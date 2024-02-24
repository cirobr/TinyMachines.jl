struct ESPNet
    ds
    encoder
    bridge
    decoder
end


function ESPNet(ch_in::Int, ch_out::Int; K=5)
    # downsample
    ds = Convk3(ch_in, 3, identity)

    # encoder
    in   = ConvK3(ch_in, 16, identity)
    esps = Chain( ESPmodule(19, 64, K; add=false),  ESPmodule(131, 128, K, add=false) )
    esp2 = Chain( ESPmodule(64, 64, K; add=true),   ESPmodule(64, 64, K; add=true) )
    esp3 = Chain( ESPmodule(128, 128, K; add=true), ESPmodule(128, 128, K; add=true), ESPmodule(128, 128, K; add=true) )

    # bridge
    bridge = Chain( ConvK1(19, ch_out, identity), ConvK1(131, ch_out, identity), ConvK1(256, ch_out, identity) )

    # decoder
    espd = ESPModule(2*ch_out, ch_out, K; add=false)
    out  = Chain( ConvK1(2*ch_out, ch_out, identity), deconv_cc )

    # output chains
    encoder = Chain(in, esps, esp2, esp3)
    decoder = Chain(espd, out)

    return ESPNet(ds, encoder, bridge, decoder)
end


function (m::ESPNet)(x)
    out1 = m.in(x)
    ds1  = m.ds(x)
    ct1 = cat(out1, ds1, dims=3)
    out2 = m.esps(ct1)
    out2 = m.esp2(out2)

    return
end