struct mobileunet
    d::Chain
    ct::Chain
    ir::Chain
end
@layer mobileunet


function mobileunet(ch_in::Int=3, ch_out::Int=2;       # input/output channels
                    activation::Function=relu6,        # activation function
                    edrops=(0.0, 0.0, 0.0, 0.0, 0.0),  # dropout rates
                    ddrops=(0.0, 0.0, 0.0, 0.0),       # dropout rates
)
    # encoder
    d1 = Chain( ConvK3(ch_in, 32, stride=2),
                BatchNorm(32, activation),
                IRBlock1(32, 16, activation, t=1),
                Dropout(edrops[1]),
    )

    d2 = Chain( IRBlock2(16, 24, activation, t=6),
                IRBlock1(24, 24, activation, t=6),
                Dropout(edrops[2]),
    )

    v3 = [IRBlock1(32, 32, activation, t=6) for _ in 1:2]
    d3 = Chain( IRBlock2(24, 32, activation, t=6),
                Chain(v3...),
                Dropout(edrops[3]),
    )

    v4a = [IRBlock1(64, 64, activation, t=6) for _ in 1:3]
    v4b = [IRBlock1(96, 96, activation, t=6) for _ in 1:2]
    d4 = Chain( IRBlock2(32, 64, activation, t=6),
                Chain(v4a...),
                IRBlock1(64, 96, activation, t=6),
                Chain(v4b...),
                Dropout(edrops[4]),
    )

    v5 = [IRBlock1(160, 160, activation, t=6) for _ in 1:2]
    d5 = Chain( IRBlock2(96, 160, activation, t=6),
                Chain(v5...),
                IRBlock1(160, 320, activation, t=6),
                ConvK1(320, 1280),
                BatchNorm(1280, activation),
                Dropout(edrops[5]),
    )

    # decoder
    ct1 = Chain(ConvTranspK4(1280, 96), BatchNorm(96, activation))
    ir1 = IRBlock1(192, 96, activation, t=1)
    ir1 = Chain(ir1, Dropout(ddrops[1]))

    ct2 = Chain(ConvTranspK4(96, 32), BatchNorm(32, activation))
    ir2 = IRBlock1(64, 32, activation, t=1)
    ir2 = Chain(ir2, Dropout(ddrops[2]))
    
    ct3 = Chain(ConvTranspK4(32, 24), BatchNorm(24, activation))
    ir3 = IRBlock1(48, 24, activation, t=1)
    ir3 = Chain(ir3, Dropout(ddrops[3]))

    ct4 = Chain(ConvTranspK4(24, 16), BatchNorm(16, activation))
    ir4 = IRBlock1(32, 16, activation, t=1)
    ir4 = Chain(ir4, Dropout(ddrops[4]))

    ct5 = ConvTranspK4(16, ch_out)

    # output chains
    d  = Chain(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5)
    ct = Chain(ct1=ct1, ct2=ct2, ct3=ct3, ct4=ct4, ct5=ct5)
    ir = Chain(ir1=ir1, ir2=ir2, ir3=ir3, ir4=ir4)

    return mobileunet(d, ct, ir)   # struct output
end

function (m::mobileunet)(x)
    # encoder
    x1 = m.d.layers.d1(x)
    x2 = m.d.layers.d2(x1)
    x3 = m.d.layers.d3(x2)
    x4 = m.d.layers.d4(x3)
    x5 = m.d.layers.d5(x4)

    # decoder
    l1 = m.ct.layers.ct1(x5)
    l2 = m.ir.layers.ir1(cat(l1, x4; dims=3))
    l3 = m.ct.layers.ct2(l2)
    l4 = m.ir.layers.ir2(cat(l3, x3; dims=3))
    l5 = m.ct.layers.ct3(l4)
    l6 = m.ir.layers.ir3(cat(l5, x2; dims=3))
    l7 = m.ct.layers.ct4(l6)
    l8 = m.ir.layers.ir4(cat(l7, x1; dims=3))
    
    logits = m.ct.layers.ct5(l8)

    # output encoder [1:5], logits[end]
    feature_maps = [x1, x2, x3, x4, x5, logits]
    return feature_maps
end


function MobileUNet(ch_in::Int=3, ch_out::Int=2;   # input/output channels
                    activation::Function=relu6,    # activation function
)
    model = mobileunet(ch_in, ch_out;
                       activation=activation,
                       edrops=(0.05, 0.05, 0.05, 0.1, 0.2),
                       ddrops=(0.0, 0.0, 0.0, 0.0),
    )
    return Chain(model, x->x[end])
end