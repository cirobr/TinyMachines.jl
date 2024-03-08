struct MobileUNet
    d::Chain
    ct::Chain
    ir::Chain
end

function MobileUNet(ch_in, ch_out)
    # encoder
    d1 = Chain(ConvK3(ch_in, 32, stride=2), BatchNorm(32, relu6),
               irblock1(32, 16, n=1, expand_ratio=1),
            #    Dropout(0.1)
    )

    d2 = Chain(irblock2(16, 24, n=2, expand_ratio=6),
            #    Dropout(0.15)
    )

    d3 = Chain(irblock2(24, 32, n=3, expand_ratio=6),
            #    Dropout(0.2)
     )

    d4 = Chain(irblock2(32, 64, n=4, expand_ratio=6),
               irblock1(64, 96, n=3, expand_ratio=6),
            #    Dropout(0.25)
    )

    d5 = Chain(irblock2(96, 160, n=3, expand_ratio=6),
               irblock1(160, 320, n=1, expand_ratio=6),
            #    Dropout(0.3),
               ConvK1(320, 1280), BatchNorm(1280, relu6)
    )


    # decoder
    ct1 = ConvTranspK4(1280, 96)
    ir1 = Chain(irblock1(192, 96, n=1, expand_ratio=1),
                # Dropout(0.25)
    )

    ct2 = ConvTranspK4(96, 32)
    ir2 = Chain(irblock1(64, 32, n=1, expand_ratio=1),
                # Dropout(0.2)
    )
    
    ct3 = ConvTranspK4(32, 24)
    ir3 = Chain(irblock1(48, 24, n=1, expand_ratio=1),
                # Dropout(0.15)
    )

    ct4 = ConvTranspK4(24, 16)
    ir4 = Chain(irblock1(32, 16, n=1, expand_ratio=1),
                # Dropout(0.1)
    )

    ct5 = ConvTranspK4(16, ch_out)


    # activation
    e0  = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)


    # output
    d  = Chain(d1, d2, d3, d4, d5)
    ct = Chain(ct1, ct2, ct3, ct4, ct5)
    ir = Chain(ir1, ir2, ir3, ir4, e0)

    return MobileUNet(d, ct, ir)
end

function (m::MobileUNet)(x)
    # x
    # encoder
    x1 = m.d[1](x)
    x2 = m.d[2](x1)
    x3 = m.d[3](x2)
    x4 = m.d[4](x3)
    x5 = m.d[5](x4)

    # decoder
    l1 = m.ct[1](x5)
    l2 = m.ir[1](cat(l1, x4; dims=3))
    l3 = m.ct[2](l2)
    l4 = m.ir[2](cat(l3, x3; dims=3))
    l5 = m.ct[3](l4)
    l6 = m.ir[3](cat(l5, x2; dims=3))
    l7 = m.ct[4](l6)
    l8 = m.ir[4](cat(l7, x1; dims=3))
    l9 = m.ct[5](l8)

    # output
    yhat = m.ir[end](l9)
    return yhat
end

@functor MobileUNet
