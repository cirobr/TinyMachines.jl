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

    # encoder: stage 1
    e1a = Chain(
        ConvK3(ch_in, 16; stride=2),
        BatchNorm(16),
        act_16,
        Dropout(edrops[1]),
    )
    # concatenation with 1st downsampled image
    e1 = Parallel( (feat,img)->cat(feat,img,dims=3), e1a, img_ds1)

    # encoder: stage 2
    e2a = ESPBlock1(19, 64; activation=activation, stride=2)
    v2b = [ESPBlock4(64, 64, activation=activation) for _ in 1:alpha2]
    e2b = Chain(v2b..., Dropout(edrops[2]))
    e2b = SkipConnection(e2b, (x,m)->cat(x,m,dims=3))
    e2c = Chain(e2a, e2b)
    # concatenation with 2nd downsampled image (last-3 channels)
    e2 = Parallel( (feat,img)->cat(feat,img,dims=3), e2c, img_ds2)

    # encoder: stage 3
    e3a = ESPBlock1(131, 128; activation=activation, stride=2)
    v3b = [ESPBlock4(128, 128, activation=activation) for _ in 1:alpha3]
    e3b = Chain(v3b..., Dropout(edrops[3]))
    e3b = SkipConnection(e3b, (x,m)->cat(x,m,dims=3))
    e3 = Chain(e3a, e3b)

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
    encoder = Chain(e1=e1, e2=e2, e3=e3)
    bridges = Chain(b1=b1, b2=b2, b3=b3)
    decoder = Chain(d2=d2, d1=d1, d0=d0)

    return espnet(encoder, bridges, decoder)   # struct output
end


function (m::espnet)(x::AbstractArray)
    # encoder (all image-fusion concatenations now live inside the layers)
    enc1 = m.encoder.layers.e1(x)
    enc2 = m.encoder.layers.e2(enc1)
    enc3 = m.encoder.layers.e3(enc2)

    # bridges
    bdg1 = m.bridges.layers.b1(enc1)
    bdg2 = m.bridges.layers.b2(enc2)
    bdg3 = m.bridges.layers.b3(enc3)

    # decoder
    dec_2 = m.decoder.layers.d2(bdg3)
    dec2 = cat(bdg2, dec_2, dims=3)

    dec_1 = m.decoder.layers.d1(dec2)
    dec1 = cat(bdg1, dec_1, dims=3)

    # logits
    return m.decoder.layers.d0(dec1)
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