struct ESPNet
    downsample
    encoder
    bridge
    decoder
end


function ESPNet(ch_in::Int, ch_out::Int; K=1, activation=relu)
    # downsample
    downsample = Chain(ConvK3(ch_in, 3, identity), BatchNorm(3), activation)


    # encoder
    inconv = Chain(ConvK3(ch_in, 16, identity),      BatchNorm(16),  activation)
    esp19  = Chain(ESPmoduleK4(19, 64; add=false),   BatchNorm(64),  activation)
    esp131 = Chain(ESPmoduleK4(131, 128; add=false), BatchNorm(128), activation)
    esp2x  = Chain(ESPmoduleK4(64, 64; add=true),    BatchNorm(64),  activation,
                   ESPmoduleK4(64, 64; add=true),    BatchNorm(64),  activation
    )
    esp3x  = Chain(ESPmoduleK4(128, 128; add=true),  BatchNorm(128), activation,
                   ESPmoduleK4(128, 128; add=true),  BatchNorm(128), activation,
                   ESPmoduleK4(128, 128; add=true),  BatchNorm(128), activation
    )


    # bridges
    bridge19  = Chain(ConvK1(19, ch_out, identity),  BatchNorm(ch_out), activation)
    bridge131 = Chain(ConvK1(131, ch_out, identity), BatchNorm(ch_out), activation)
    bridge256 = Chain(ConvK1(256, ch_out, identity), BatchNorm(ch_out), activation)


    # decoder
    deconv  = Chain(ConvTranspK2(ch_out, ch_out, identity; stride=1), BatchNorm(ch_out), activation)
    espdec  = Chain(ESPmoduleK1(2*ch_out, ch_out; add=false),         BatchNorm(ch_out), activation)
    outconv = ConvK1(2*ch_out, ch_out, identity)
    e0 = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)


    # output chains
    encoder = Chain(inconv=inconv, esp19=esp19, esp131=esp131, esp2x=esp2x, esp3x=esp3x)
    bridge  = Chain(bridge19=bridge19, bridge131=bridge131, bridge256=bridge256)
    decoder = Chain(deconv=deconv, espdec=espdec, outconv=outconv, e0=e0)

    return ESPNet(downsample, encoder, bridge, decoder)
end


function (m::ESPNet)(x)
    # encoder
    x
    # out1 = m.encoder[:inconv](x)
    # ds1  = m.downsample(x)
    # ct1 = cat(ds1, out1, dims=3)

    # out2 = m.encoder[:esp19](ct1)
    # out3 = m.encoder[:esp2x](out2)
    # ct2 = cat(ds1, out2, out3, dims=3)
    
    # out4 = m.encoder[:esp131](ct2)
    # out5 = m.encoder[:esp3x](out4)
    # ct3 = cat(out4, out5, dims=3)

    # # bridges
    # b19 = m.bridge[:bridge19](ct1)
    # b131 = m.bridge[:bridge131](ct2)
    # b256 = m.bridge[:bridge256](ct3)

    # # decoder
    # out6 = m.decoder[:deconv](b256)
    # ct4 = cat(b131, out6, dims=3)

    # out7 = m.decoder[:espdec](ct4)
    # out8 = m.decoder[:deconv](out7)
    # ct5 = cat(b19, out8, dims=3)

    # out9 = m.decoder[:outconv](ct5)
    # out10 = m.decoder[:deconv](out9)
    # yhat = m.decoder[:e0](out10)

    # return yhat
end

Flux.@functor ESPNet
