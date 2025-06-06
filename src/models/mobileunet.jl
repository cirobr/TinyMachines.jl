struct mobileunet
    d::Chain
    ct::Chain
    ir::Chain
end
@layer mobileunet


function mobileunet(ch_in::Int=3, ch_out::Int=1;   # input/output channels
                    activation::Function=relu6,    # activation function
                    edrops=(0.0, 0.0, 0.0, 0.0, 0.0), # dropout rates
                    ddrops=(0.0, 0.0, 0.0, 0.0),     # dropout rates
                    # verbose::Bool = false,         # output feature maps
)
    # encoder
    d1 = Chain( ConvK3(ch_in, 32, stride=2), BatchNorm(32, activation),
                BBlock(32, 16, activation, stride=1, expansion_factor=1, n=1),
                Dropout(edrops[1]),   # Dropout(0.05)
    )
    d2 = Chain( BBlock(16, 24, activation, stride=2, expansion_factor=6, n=2),
                Dropout(edrops[2]),   # Dropout(0.05)
    )
    d3 = Chain( BBlock(24, 32, activation, stride=2, expansion_factor=6, n=3),
                Dropout(edrops[3]),   # Dropout(0.05)
    )
    d4 = Chain( BBlock(32, 64, activation, stride=2, expansion_factor=6, n=4),
                BBlock(64, 96, activation, stride=1, expansion_factor=6, n=3),
                Dropout(edrops[4]),   # Dropout(0.1)
    )
    d5 = Chain( BBlock(96, 160, activation, stride=2, expansion_factor=6, n=3),
                BBlock(160, 320, activation, stride=1, expansion_factor=6, n=1),
                ConvK1(320, 1280), BatchNorm(1280, activation),
                Dropout(edrops[5]),   # Dropout(0.2)
    )

    # decoder
    ct1 = Chain(ConvTranspK4(1280, 96), BatchNorm(96, activation))
    ir1 = BBlock(192, 96, activation, stride=1, expansion_factor=1, n=1)
    ir1 = Chain(ir1, Dropout(ddrops[1]))

    ct2 = Chain(ConvTranspK4(96, 32), BatchNorm(32, activation))
    ir2 = BBlock(64, 32, activation, stride=1, expansion_factor=1, n=1)
    ir2 = Chain(ir2, Dropout(ddrops[2]))
    
    ct3 = Chain(ConvTranspK4(32, 24), BatchNorm(24, activation))
    ir3 = BBlock(48, 24, activation, stride=1, expansion_factor=1, n=1)
    ir3 = Chain(ir3, Dropout(ddrops[3]))

    ct4 = Chain(ConvTranspK4(24, 16), BatchNorm(16, activation))
    ir4 = BBlock(32, 16, activation, stride=1, expansion_factor=1, n=1)
    ir4 = Chain(ir4, Dropout(ddrops[4]))

    ct5 = ConvTranspK4(16, ch_out)

    # output chains
    d  = Chain(d1, d2, d3, d4, d5)
    ct = Chain(ct1, ct2, ct3, ct4, ct5)
    ir = Chain(ir1, ir2, ir3, ir4)

    return mobileunet(d, ct, ir)   # struct output
end

function (m::mobileunet)(x)
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
    logits = l9

    # output
    feature_maps = [x1, x2, x3, x4, x5,                       # encoder [1:5]
                    l1, l2, l3, l4, l5, l6, l7, l8, logits]   # decoder [6:14]
    return feature_maps
end


function MobileUNet(ch_in::Int=3, ch_out::Int=1;   # input/output channels
                    activation::Function=relu6,    # activation function
)
    model = mobileunet(ch_in, ch_out;
                       activation=activation,
                       edrops=(0.05, 0.05, 0.05, 0.1, 0.2),
                       ddrops=(0.0, 0.0, 0.0, 0.0),
    )
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)
    return Chain(model, x->x[end], act)
end