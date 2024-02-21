struct ESPNet
    encoder::Chain
    bridge::Chain
    decoder::Chain
end


function ESPNet(ch_in::Int=3, ch_out::Int=1)
    # encoder & bridge
    e1 = Chain(ConvK3(ch_in, 16), BatchNorm(16, prelu2))
    # concat(x, e1, dims=3) 19 channels out
    br1 = ConvK1(19, ch_out)
    # e2 = Chain(ESPmodule(19, 64), ESPmodule(64, 64; α=2))
    # concat(x, e2[1], e2[2], dims=3) 131 channels out
    br2 = ConvK1(131, ch_out)
    # e3 = Chain(ESPmodule(131, 128), ESPmodule(128, 128; α=3))
    # concat(e3[1], e3[2]) 256 channels out
    br3 = ConvK1(256, ch_out)

    # decoder



    # output
    encoder = Chain(e1, e2, e3)
    bridge  = Chain(br1, br2, br3)
    # decoder = 

    return ESPNet(encoder, bridge, decoder)
end


function (m::ESPNet)(x::Array{Float32, 4})

    return yhat
end


