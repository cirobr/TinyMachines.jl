concat(x,y) = cat(x,y;dims=3)


function unet5_aux(ch_in::Int=3, ch_out::Int=2;      # input/output channels
               activation::Function = relu,          # activation function
               alpha::Int           = 1,             # channels divider
               edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
               ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)

    chs = defaultChannels .÷ alpha

    # encoder
    e1 = Chain(CB(ch_in, chs[1], activation),   Dropout(edrops[1]))
    e2 = Chain(MCB(chs[1], chs[2], activation), Dropout(edrops[2]))
    e3 = Chain(MCB(chs[2], chs[3], activation), Dropout(edrops[3]))
    e4 = Chain(MCB(chs[3], chs[4], activation), Dropout(edrops[4]))
    e5 = Chain(MCB(chs[4], chs[5], activation), Dropout(edrops[5]))

    # up convolutions
    u4 = ConvTranspK2(chs[5], chs[4], activation; stride=2)
    u3 = ConvTranspK2(chs[4], chs[3], activation; stride=2)
    u2 = ConvTranspK2(chs[3], chs[2], activation; stride=2)
    u1 = ConvTranspK2(chs[2], chs[1], activation; stride=2)

    # decoder
    d4 = Chain(CB(chs[5], chs[4], activation), Dropout(ddrops[4]))
    d3 = Chain(CB(chs[4], chs[3], activation), Dropout(ddrops[3]))
    d2 = Chain(CB(chs[3], chs[2], activation), Dropout(ddrops[2]))
    d1 = Chain(CB(chs[2], chs[1], activation), Dropout(ddrops[1]))
    d0 = ConvK1(chs[1], ch_out)

    # model
    l5 = Chain(e5, u4)
    l4 = Chain(e4, SkipConnection(l5, concat), d4, u3)
    l3 = Chain(e3, SkipConnection(l4, concat), d3, u2)
    l2 = Chain(e2, SkipConnection(l3, concat), d2, u1)
    logits = Chain(e1, SkipConnection(l2, concat), d1, d0)

    return logits
end
