struct MobUNet
    d::Chain
    ct::Chain
    ir::Chain
end

function MobUNet(ch_in, ch_out)
    # encoder
    d1 = Chain(ConvK3(ch_in, 32, stride=2), BatchNorm(32, relu6),
               irblocks(32, 16, n=1, stride=1, expand_ratio=1),
               Dropout(0.1)
    )

    d2 = Chain(irblocks(16, 24, n=2, stride=2, expand_ratio=6),
               Dropout(0.15)
    )

    d3 = Chain(irblocks(24, 32, n=3, stride=2, expand_ratio=6),
               Dropout(0.2)
     )

    d4 = Chain(irblocks(32, 64, n=4, stride=2, expand_ratio=6),
               irblocks(64, 96, n=3, stride=1, expand_ratio=6),
               Dropout(0.25)
    )

    d5 = Chain(irblocks(96, 160, n=3, stride=2, expand_ratio=6),
               irblocks(160, 320, n=1, stride=1, expand_ratio=6),
               Dropout(0.3),
               ConvK1(320, 1280), BatchNorm(1280, relu6)
    )


    # decoder
    ct1 = ConvTranspK4(1280, 96)
    ir1 = Chain(irblocks(192, 96, n=1, stride=1, expand_ratio=1),
                Dropout(0.25)
    )

    ct2 = ConvTranspK4(96, 32)
    ir2 = Chain(irblocks(64, 32, n=1, stride=1, expand_ratio=1),
                Dropout(0.2)
    )
    
    ct3 = ConvTranspK4(32, 24)
    ir3 = Chain(irblocks(48, 24, n=1, stride=1, expand_ratio=1),
                Dropout(0.15)
    )

    ct4 = ConvTranspK4(24, 16)
    ir4 = Chain(irblocks(32, 16, n=1, stride=1, expand_ratio=1),
                Dropout(0.1)
    )

    ct5 = ConvTranspK4(16, ch_out)


    # activation
    e0  = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)


    # output
    d  = Chain(d1, d2, d3, d4, d5)
    ct = Chain(ct1, ct2, ct3, ct4, ct5)
    ir = Chain(ir1, ir2, ir3, ir4, e0)

    return MobUNet(d, ct, ir)
end

function (m::MobUNet)(x)
    # encoder
    x1 = m.d[1](x)
    x2 = m.d[2](x1)
    x3 = m.d[3](x2)
    x4 = m.d[4](x3)
    x5 = m.d[5](x4)

    # decoder
    l1 = m.ct[1](x5)
    # size(l1), size(x4)
    l2 = m.ir[1](cat(l1, x4; dims=3))
    l3 = m.ct[2](l2)
    # size(l3), size(x3)
    l4 = m.ir[2](cat(l3, x3; dims=3))
    l5 = m.ct[3](l4)
    # size(l5), size(x2)
    l6 = m.ir[3](cat(l5, x2; dims=3))
    l7 = m.ct[4](l6)
    l8 = m.ir[4](cat(l7, x1; dims=3))

    # output
    l9 = m.ct[5](l8)
    return m.ir[5](l9)
end

@functor MobUNet
