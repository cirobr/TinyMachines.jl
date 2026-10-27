struct mobileunet
    encoder::Chain          # d1…d5 (no dropouts inside)
    upconvs::Chain          # ct1…ct5
    decoder::Chain          # ir1…ir4
    eds::Chain              # encoder dropouts
    dds::Chain              # decoder dropouts
end
@layer mobileunet trainable=(encoder, upconvs, decoder)

function mobileunet(
    ch_in::Int = 3,                       # input channels
    ch_out::Int = 2;                      # output channels
    t::Int = 6,                           # expansion factor (t=6 follows paper)
    activation::Function = relu6,         # activation function
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # encoder dropouts
    ddrops = (0.0, 0.0, 0.0, 0.0),        # decoder dropouts
)
    # encoder dropouts
    ed1 = Dropout(edrops[1])
    ed2 = Dropout(edrops[2])
    ed3 = Dropout(edrops[3])
    ed4 = Dropout(edrops[4])
    ed5 = Dropout(edrops[5])

    # decoder dropouts
    dd1 = Dropout(ddrops[1])
    dd2 = Dropout(ddrops[2])
    dd3 = Dropout(ddrops[3])
    dd4 = Dropout(ddrops[4])

    # ── encoder (MobileNetV2 feature extractor) ──────────────
    d1 = Chain(
        Conv3(ch_in, 32; stride=2),
        BatchNorm(32, activation),
        IRBlock1(32, 16, activation; t=1),
    )

    d2 = Chain(
        IRBlock2(16, 24, activation; t=t),
        IRBlock1(24, 24, activation; t=t),
    )

    d3 = Chain(
        IRBlock2(24, 32, activation; t=t),
        Chain([IRBlock1(32, 32, activation; t=t) for _ in 1:2]...),
    )

    d4 = Chain(
        IRBlock2(32, 64, activation; t=t),
        Chain([IRBlock1(64, 64, activation; t=t) for _ in 1:3]...),
        IRBlock1(64, 96, activation; t=t),
        Chain([IRBlock1(96, 96, activation; t=t) for _ in 1:2]...),
    )

    d5 = Chain(
        IRBlock2(96, 160, activation; t=t),
        Chain([IRBlock1(160, 160, activation; t=t) for _ in 1:2]...),
        IRBlock1(160, 320, activation; t=t),
        Conv1(320, 1280),
        BatchNorm(1280, activation),
    )

    # ── up-convolutions (paper: k=4, s=2) ────────────────────
    ct1 = Chain(ConvTr4(1280, 96), BatchNorm(96, activation))
    ct2 = Chain(ConvTr4(96,   32), BatchNorm(32, activation))
    ct3 = Chain(ConvTr4(32,   24), BatchNorm(24, activation))
    ct4 = Chain(ConvTr4(24,   16), BatchNorm(16, activation))
    ct5 = ConvTr4(16, ch_out)

    # ── decoder inverted residuals (paper: t=6) ──────────────
    # after cat: 96+96→192, 32+32→64, 24+24→48, 16+16→32
    ir1 = IRBlock1(192, 96, activation; t=t)
    ir2 = IRBlock1(64,  32, activation; t=t)
    ir3 = IRBlock1(48,  24, activation; t=t)
    ir4 = IRBlock1(32,  16, activation; t=t)

    # pack into named Chains (unet5 style)
    encoder = Chain(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5)
    upconvs = Chain(ct1=ct1, ct2=ct2, ct3=ct3, ct4=ct4, ct5=ct5)
    decoder = Chain(ir1=ir1, ir2=ir2, ir3=ir3, ir4=ir4)
    eds     = Chain(ed1=ed1, ed2=ed2, ed3=ed3, ed4=ed4, ed5=ed5)
    dds     = Chain(dd1=dd1, dd2=dd2, dd3=dd3, dd4=dd4)

    return mobileunet(encoder, upconvs, decoder, eds, dds)
end


function (m::mobileunet)(x)
    # encoder (drops applied only on continuing path)
    x1  = m.encoder.layers.d1(x)
    x1d = m.eds.layers.ed1(x1)
    x2  = m.encoder.layers.d2(x1d)
    x2d = m.eds.layers.ed2(x2)
    x3  = m.encoder.layers.d3(x2d)
    x3d = m.eds.layers.ed3(x3)
    x4  = m.encoder.layers.d4(x3d)
    x4d = m.eds.layers.ed4(x4)
    x5  = m.encoder.layers.d5(x4d)
    x5d = m.eds.layers.ed5(x5)

    # decoder
    l1  = m.upconvs.layers.ct1(x5d)
    l2  = m.decoder.layers.ir1(cat(l1, x4; dims=3))
    l2d = m.dds.layers.dd1(l2)

    l3  = m.upconvs.layers.ct2(l2d)
    l4  = m.decoder.layers.ir2(cat(l3, x3; dims=3))
    l4d = m.dds.layers.dd2(l4)

    l5  = m.upconvs.layers.ct3(l4d)
    l6  = m.decoder.layers.ir3(cat(l5, x2; dims=3))
    l6d = m.dds.layers.dd3(l6)

    l7  = m.upconvs.layers.ct4(l6d)
    l8  = m.decoder.layers.ir4(cat(l7, x1; dims=3))
    l8d = m.dds.layers.dd4(l8)

    # logits
    return m.upconvs.layers.ct5(l8d)
end


# paper-faithful:    edrops=(0.0, 0.0, 0.0, 0.0, 0.0), ddrops=(0.0, 0.0, 0.0, 0.0)
# mild overfitting:  edrops=(0.05, 0.05, 0.05, 0.1, 0.2), ddrops=(0.0, 0.0, 0.0, 0.0)
# clear overfitting: edrops=(0.1, 0.1, 0.15, 0.2, 0.3),  ddrops=(0.05, 0.05, 0.1, 0.1)
function MobileUNet(
    ch_in::Int = 3,
    ch_out::Int = 2;
    activation::Function = relu6,
)
    return mobileunet(
        ch_in, ch_out;
        activation = activation,
        t = 6,
        edrops = (0.0, 0.0, 0.0, 0.0, 0.0),
        ddrops = (0.0, 0.0, 0.0, 0.0),
    )
end