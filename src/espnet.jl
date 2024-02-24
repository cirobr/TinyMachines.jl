struct ESPNet
    downsample
    encoder
    bridge
    decoder
end


function ESPNet(ch_in::Int, ch_out::Int; K=5)
    # downsample
    downsample = Convk3(ch_in, 3, identity)


    # encoder
    in     = ConvK3(ch_in, 16, identity)
    esp19  = ESPmodule(19, 64; K=K, add=false)
    esp131 = ESPmodule(131, 128; K=K, add=false)

    esp2x  = Chain(ESPmodule(64, 64; K=K, add=true),
                   ESPmodule(64, 64; K=K, add=true)
    )
    esp3x  = Chain(ESPmodule(128, 128; K=K, add=true),
                   ESPmodule(128, 128; K=K, add=true),
                   ESPmodule(128, 128; K=K, add=true)
    )


    # bridges
    bridge19  = ConvK1(19, ch_out, identity)
    bridge131 = ConvK1(131, ch_out, identity)
    bridge256 = ConvK1(256, ch_out, identity)


    # decoder
    ### deconv_cc
    espdec = ESPModule(2*ch_out, ch_out; K=K, add=false)
    out    = Chain(ConvK1(2*ch_out, ch_out, identity), deconv_cc)


    # output chains
    encoder = Chain(in=in, esp19=esp19, esp131=esp131, esp2x=esp2x, esp3x=esp3x)
    bridge  = Chain(bridge19=bridge19, bridge131=bridge131, bridge256=bridge256)
    decoder = Chain(espdec=espdec, out=out)

    return ESPNet(downsample, encoder, bridge, decoder)
end


function (m::ESPNet)(x)
    # encoder
    out1 = m.encoder[:in](x)
    ds1  = m.downsample(x)
    ct1 = cat(ds1, out1, dims=3)
    out2 = m.encoder[:esp19](ct1)
    out3 = m.encoder[:esp2x](out2)
    ct2 = cat(ds1, out2, out3, dims=3)
    out4 = m.encoder[:esp131](ct2)
    out5 = m.encoder[:esp3x](out4)
    ct3 = cat(out4, out5, dims=3)

    # bridges
    b19 = m.bridge[:bridge19](ct1)
    b131 = m.bridge[:bridge131](ct2)
    b256 = m.bridge[:bridge256](ct3)

    # decoder
    

    return
end