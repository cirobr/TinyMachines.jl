struct espnet
    # encoder
    l1::Chain
    l2_0::DownSamplerB
    l2::Chain
    l3_0::DownSamplerB
    l3::Chain
    fusion1::Chain
    fusion2::Chain
    fusion3::Chain
    # skip connections
    b1::Conv
    b2::Conv
    b3::Conv
    # decoder
    level3_C::Conv
    deep_C::Conv
    br::BatchNorm
    up_l3::ConvTranspose
    combine::Chain
    up_l2::Chain
    dec_conv::Chain
    final_up::ConvTranspose
    # dropouts (non-trainable)
    eds::Chain
    dds::Chain
end
@layer espnet trainable=(l1, l2_0, l2, l3_0, l3, fusion1, fusion2, fusion3, b1, b2, b3,
                         level3_C, deep_C, br, up_l3, combine, up_l2, dec_conv, final_up)


function espnet(
    ch_in::Int=3,             # input channels
    ch_out::Int=2;            # output channels
    activation = "prelu",     # activation function
    alpha2::Int=2,            # repeats for level 2 (p in original)
    alpha3::Int=3,            # repeats for level 3 (q in original)
    edrops=(0.0, 0.0, 0.0),   # dropout rates for encoder main path
    ddrops=(0.0, 0.0),        # dropout rates for decoder main path
)
    # activations
    act_ch_out = (activation == "prelu" ? PReLU(ch_out) : activation)
    act_16     = (activation == "prelu" ? PReLU(16) : activation)
    act_19     = (activation == "prelu" ? PReLU(19) : activation)
    act_131    = (activation == "prelu" ? PReLU(131) : activation)
    act_256    = (activation == "prelu" ? PReLU(256) : activation)
    act_2c     = (activation == "prelu" ? PReLU(2*ch_out) : activation)

    # dropouts (applied only on main continuing path, not on fused "skip" features)
    ed1 = Dropout(edrops[1])
    ed2 = Dropout(edrops[2])
    ed3 = Dropout(edrops[3])

    dd2 = Dropout(ddrops[2])
    dd1 = Dropout(ddrops[1])

    # encoder
    l1 = Chain(
        ConvK3(ch_in, 16; stride=2),
        BatchNorm(16),
        act_16,
    )

    l2_0 = DownSamplerB(19, 64; activation=activation)
    v2   = [DilatedParallelResidualBlockB(64, 64; add=true, activation=activation)
            for _ in 1:alpha2]
    l2   = Chain(v2...)

    l3_0 = DownSamplerB(131, 128; activation=activation)
    v3   = [DilatedParallelResidualBlockB(128, 128; add=true, activation=activation)
            for _ in 1:alpha3]
    l3   = Chain(v3...)

    # fusion BRs (BN + act after cat, matching original b1/b2/b3 style before bridges)
    fusion1 = Chain(BatchNorm(19),  act_19)
    fusion2 = Chain(BatchNorm(131), act_131)
    fusion3 = Chain(BatchNorm(256), act_256)

    # bridges (1x1 projections of the fused features)
    b1 = ConvK1(19,  ch_out)
    b2 = ConvK1(131, ch_out)
    b3 = ConvK1(256, ch_out)

    # === Decoder (light-weight RUM style from original) ===
    level3_C = ConvK1(131, ch_out)
    deep_C   = ConvK1(256, ch_out)   # for output2_cat projection
    br       = BatchNorm(ch_out)
    up_l3    = ConvTrK2(ch_out, ch_out)

    combine = if ch_out < 5
        Chain(
            BatchNorm(2 * ch_out),
            act_2c,
            ConvK1(2*ch_out, ch_out),
            ConvK3(ch_out, ch_out),
            BatchNorm(ch_out),
            act_ch_out,
        )
    else
        Chain(
            BatchNorm(2 * ch_out),
            act_2c,
            DilatedParallelResidualBlockB(2 * ch_out, ch_out; add=false, activation=activation),
        )
    end

    up_l2 = Chain(
        ConvTrK2(ch_out, ch_out),
        BatchNorm(ch_out),
        act_ch_out,
    )

    dec_conv = Chain(
        ConvK3(19 + ch_out, ch_out),
        BatchNorm(ch_out),
        act_ch_out,
    )

    final_up = ConvTrK2(ch_out, ch_out)

    # pack dropouts
    eds = Chain(ed1=ed1, ed2=ed2, ed3=ed3)
    dds = Chain(dd2=dd2, dd1=dd1)

    return espnet(l1, l2_0, l2, l3_0, l3,
                  fusion1, fusion2, fusion3,
                  b1, b2, b3,
                  level3_C, deep_C, br, up_l3, combine, up_l2, dec_conv, final_up,
                  eds, dds)
end


function (m::espnet)(x::AbstractArray)
    # === Encoder (replicates original forward + fusion points) ===
    output0   = m.l1(x)
    inp1      = img_ds1(x)
    fused0    = cat(output0, inp1; dims=3)
    output0_cat = m.fusion1(fused0)   # BR(19) style

    # ed1 only on main continuing path (skip/fusion features stay pre-drop)
    output0d  = m.eds.layers.ed1(output0_cat)

    output1_0 = m.l2_0(output0d)
    output1   = output1_0
    for layer in m.l2
        output1 = layer(output1)
    end

    inp2 = img_ds2(x)                    # always project the raw input image (original behavior)
    fused1    = cat(output1, output1_0, inp2; dims=3)
    output1_cat = m.fusion2(fused1)

    # ed2 only on main
    output1d = m.eds.layers.ed2(output1_cat)

    output2_0 = m.l3_0(output1d)
    output2   = output2_0
    for layer in m.l3
        output2 = layer(output2)
    end

    fused2    = cat(output2_0, output2; dims=3)
    output2_cat = m.fusion3(fused2)

    # ed3 only on main deep path
    output2d = m.eds.layers.ed3(output2_cat)

    # bridges use the pre-dropout fused features (philosophy from unets)
    bdg1 = m.b1(output0_cat)
    bdg2 = m.b2(output1_cat)
    bdg3 = m.b3(output2d)

    # === Decoder (RUM style from original Python) ===
    # deep projection + up
    deep_p = m.br( m.deep_C(output2_cat) )
    output2_c = m.up_l3(deep_p)

    # mid projection
    output1_C = m.level3_C(output1_cat)

    # combine
    comb_in = cat(output1_C, output2_c; dims=3)
    comb    = m.combine(comb_in)
    comb_up = m.up_l2(comb)

    # concat with early fused + conv
    concat  = cat(comb_up, output0_cat; dims=3)
    dec_out = m.dec_conv(concat)

    # final logits
    return m.final_up(dec_out)
end


function ESPNet(
    ch_in::Int=3,
    ch_out::Int=2;
    activation="prelu"
)
    return espnet(
        ch_in,
        ch_out;
        activation=activation,
        alpha2=2,          # matches frequently cited paper value (α2=2)
        alpha3=8,          # matches frequently cited paper value (α3=8)
        edrops=(0.05, 0.1, 0.2),
        ddrops=(0.05, 0.1),
    )
end
