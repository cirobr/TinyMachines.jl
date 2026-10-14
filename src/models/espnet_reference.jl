struct espnet
    # image downsamplers
    ds1::MeanPool
    ds2::Chain
    # encoder
    e1::Chain
    e2_down::DownSampler
    e2_deep::Chain
    e3_down::DownSampler
    e3_deep::Chain
    #fusion bridges
    fusion1::Chain
    fusion2::Chain
    fusion3::Chain
    # bridges / skip connections
    bridge1::Conv
    bridge2::Conv
    bridge3::Conv
    # decoder
    d3_up::ConvTranspose
    d2_deep::Chain
    d2_up::Chain
    d1_conv::Chain
    d1_up::ConvTranspose
    # dropouts (non-trainable)
    eds::Chain
    dds::Chain
end
@layer espnet trainable=(
    ds1,ds2,
    e1,e2_down,e2_deep,e3_down,e3_deep,
    fusion1,fusion2,fusion3,bridge1,bridge2,bridge3,
    d3_up,d2_deep,d2_up,d1_conv,
)


function espnet(
    ch_in::Int=3,             # input channels
    ch_out::Int=2;            # output channels
    activation = "prelu",     # activation function
    alpha2::Int=2,            # repeats for level 2 (p in original)
    alpha3::Int=3,            # repeats for level 3 (q in original)
    edrops=(0.0, 0.0, 0.0),   # dropout rates for encoder main path
    ddrops=(0.0, 0.0),        # dropout rates for decoder main path
)
    # dropouts (applied only on main continuing path, not on fused "skip" features)
    edrop1 = Dropout(edrops[1])
    edrop2 = Dropout(edrops[2])
    edrop3 = Dropout(edrops[3])

    ddrop2 = Dropout(ddrops[2])
    ddrop1 = Dropout(ddrops[1])

    # image downsamplers
    ds1 = MeanPool((3,3); stride=2, pad=1)
    ds2 = Chain(MeanPool((3,3); stride=2, pad=1),
                MeanPool((3,3); stride=2, pad=1)
    )

    # activations
    act_ch_out = (activation == "prelu" ? PReLU(ch_out) : activation)
    act_16     = (activation == "prelu" ? PReLU(16) : activation)
    act_19     = (activation == "prelu" ? PReLU(19) : activation)
    act_131    = (activation == "prelu" ? PReLU(131) : activation)
    act_256    = (activation == "prelu" ? PReLU(256) : activation)
    act_2c     = (activation == "prelu" ? PReLU(2*ch_out) : activation)

    # encoder
    e1 = Chain(
        ConvK3(ch_in, 16; stride=2),
        BatchNorm(16),
        act_16,
    )

    e2_down = DownSampler(19, 64; activation=activation)
    v2      = [ESPModule(64, 64; add=true, activation=activation)
               for _ in 1:alpha2]
    e2_deep = Chain(v2...)

    e3_down = DownSampler(131, 128; activation=activation)
    v3      = [ESPModule(128, 128; add=true, activation=activation)
               for _ in 1:alpha3]
    e3_deep = Chain(v3...)

    # fusion bridges (BN + acttivation after cat)
    fusion1 = Chain(BatchNorm(19),  act_19)
    fusion2 = Chain(BatchNorm(131), act_131)
    fusion3 = Chain(BatchNorm(256), act_256)

    # bridges (1x1 projections of the fused features)
    bridge1 = ConvK1(19,  ch_out)
    bridge2 = ConvK1(131, ch_out)
    bridge3 = ConvK1(256, ch_out)

    # decoder
    d3_up    = ConvTrK2(ch_out, ch_out)

    d2_deep = if ch_out < 5
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
            ESPModule(2 * ch_out, ch_out; add=false, activation=activation),
        )
    end

    d2_up = Chain(
        ConvTrK2(ch_out, ch_out),
        BatchNorm(ch_out),
        act_ch_out,
    )

    d1_conv = Chain(
        ConvK1(2 * ch_out, ch_out),
        BatchNorm(ch_out),
        act_ch_out,
    )

    d1_up = ConvTrK2(ch_out, ch_out)

    # pack dropouts
    eds = Chain(edrop1=edrop1, edrop2=edrop2, edrop3=edrop3)
    dds = Chain(ddrop2=ddrop2, ddrop1=ddrop1)

    return espnet(ds1, ds2,
                  e1, e2_down, e2_deep, e3_down, e3_deep,
                  fusion1, fusion2, fusion3,
                  bridge1, bridge2, bridge3,
                  d3_up, d2_deep, d2_up, d1_conv, d1_up,
                  eds, dds)
end


function (m::espnet)(x::AbstractArray)
    # encoder
    e1_out      = m.e1(x)
    inp1        = m.ds1(x)
    e1_out_cat  = cat(e1_out, inp1; dims=3)
    e1_fused    = m.fusion1(e1_out_cat)
    e1_fused_drop = m.eds.layers.edrop1(e1_fused)

    e2_down_out = m.e2_down(e1_fused_drop)
    e2_deep_out = m.e2_deep(e2_down_out)
    inp2        = m.ds2(x)
    e2_out_cat  = cat(e2_down_out, e2_deep_out, inp2; dims=3)
    e2_fused    = m.fusion2(e2_out_cat)
    e2_fused_drop = m.eds.layers.edrop2(e2_fused)

    e3_down_out = m.e3_down(e2_fused_drop)
    e3_deep_out = m.e3_deep(e3_down_out)
    e3_out_cat  = cat(e3_down_out, e3_deep_out; dims=3)
    e3_fused    = m.fusion3(e3_out_cat)
    e3_fused_drop = m.eds.layers.edrop3(e3_fused)

    # bridges
    b3_out = m.bridge3(e3_fused_drop)   # dropout applied at main flow
    b2_out = m.bridge2(e2_fused)
    b1_out = m.bridge1(e1_fused)

    # # decoder
    d3_up_out = m.d3_up(b3_out)

    d2_out_cat = cat(b2_out, d3_up_out; dims=3)
    d2_out_cat_drop = m.dds.layers.ddrop2(d2_out_cat)
    d2_deep_out = m.d2_deep(d2_out_cat_drop)
    d2_up_out = m.d2_up(d2_deep_out)

    d1_out_cat = cat(b1_out, d2_up_out; dims=3)
    d1_out_cat_drop = m.dds.layers.ddrop1(d1_out_cat)
    d1_conv_out = m.d1_conv(d1_out_cat_drop)
    
    # logits
    return m.d1_up(d1_conv_out)
end

# paper-faithful:    edrops=(0.0, 0.0, 0.0), ddrops=(0.0, 0.0)
# mild overfitting:  edrops=(0.01, 0.05, 0.1), ddrops=(0.05, 0.1)
# clear overfitting: edrops=(0.05, 0.1, 0.2), ddrops=(0.1, 0.15)
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
        edrops=(0.0, 0.0, 0.0),
        ddrops=(0.0, 0.0),
    )
end
