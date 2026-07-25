struct espnet
    encoder::Chain
    bridges::Chain
    decoder::Chain
end
@layer espnet


function espnet(
    ch_in::Int=3,             # input channels
    ch_out::Int=2;            # output channels
    activation = "prelu",     # activation function
    alpha2::Int=2,            # expansion factor in encoder stage 2
    alpha3::Int=3,            # expansion factor in encoder stage 3
    edrops=(0.0, 0.0, 0.0),   # dropout rates for encoder
    ddrops=(0.0, 0.0),        # dropout rates for decoder
)
    # activations
    act_16     = ( activation == "prelu" ? PReLU(16) : activation )
    act_ch_out = ( activation == "prelu" ? PReLU(ch_out) : activation )

    # encoder
    e1  = Chain(
        ConvK3(ch_in, 16; stride=2),
        BatchNorm(16),
        act_16,
        Dropout(edrops[1]),
    )

    e2a = ESPBlock1(19, 64; activation=activation, stride=2)
    
    v2b = [ESPBlock4(64, 64, activation=activation) for _ in 1:alpha2]
    e2b = Chain(v2b..., Dropout(edrops[2]))

    e3a = ESPBlock1(131, 128; activation=activation, stride=2)
    v3b = [ESPBlock4(128, 128, activation=activation) for _ in 1:alpha3]
    e3b = Chain(v3b..., Dropout(edrops[3]))

    # bridges
    b1 = ConvK1(19,  ch_out)
    b2 = ConvK1(131, ch_out)
    b3 = ConvK1(256, ch_out)

    # decoder
    d2 = Chain(
        ConvTrK2(ch_out, ch_out; stride=2),
        BatchNorm(ch_out),
        act_ch_out,
        Dropout(ddrops[2]),
    )

    d1 = Chain(
        ESPBlock1(2*ch_out, ch_out; activation=activation, stride=1),
        ConvTrK2(ch_out, ch_out; stride=2),
        BatchNorm(ch_out),
        act_ch_out,
        Dropout(ddrops[1]),
    )
    
    d0 = Chain(
        ConvK1(2*ch_out, ch_out),
        ConvTrK2(ch_out, ch_out; stride=2),   # no bn, no activation
    )

    # output chains
    encoder = Chain(e1=e1, e2a=e2a, e2b=e2b, e3a=e3a, e3b=e3b)
    bridges = Chain(b1=b1, b2=b2, b3=b3)
    decoder = Chain(d2=d2, d1=d1, d0=d0)

    return espnet(encoder, bridges, decoder)   # struct output
end


function (m::espnet)(x::AbstractArray; return_features::Bool = false)
    # input image downsampling
    x1 = downsampling(x)
    x2 = downsampling(x1)

    # encoder
    f1 = m.encoder.layers.e1(x)
    enc1 = cat(x1, f1, dims=3)

    f2 = m.encoder.layers.e2a(enc1)
    f3 = m.encoder.layers.e2b(f2)
    enc2 = cat(x2, f2, f3, dims=3)
    
    f4 = m.encoder.layers.e3a(enc2)
    f5 = m.encoder.layers.e3b(f4)
    enc3 = cat(f4, f5, dims=3)


    # bridges
    b1 = m.bridges.layers.b1(enc1)
    b2 = m.bridges.layers.b2(enc2)
    b3 = m.bridges.layers.b3(enc3)


    # decoder
    d2 = m.decoder.layers.d2(b3)
    dec2 = cat(b2, d2, dims=3)

    d1 = m.decoder.layers.d1(dec2)
    dec1 = cat(b1, d1, dims=3)

    logits = m.decoder.layers.d0(dec1)

    # output features, logits
    if return_features
        return (logits  = logits,
                encoder = (enc1=enc1, enc2=enc2, enc3=enc3)
        )
    else
        return logits
    end
end


function ESPNet(
    ch_in::Int=3,
    ch_out::Int=2;
    activation="prelu"
)
    return espnet(
        ch_in,
        ch_out;
        activation=activation,
        alpha2=5,
        alpha3=8,
        edrops=(0.0, 0.1, 0.3),
        ddrops=(0.0, 0.0),
    )
end