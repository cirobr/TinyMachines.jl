struct espnet
    encoder::Chain
    bridges::Chain
    decoder::Chain
end
@layer espnet


# PReLU is incorporated, no need to pass activation function
function espnet(ch_in::Int=3, ch_out::Int=1;   # input/output channels
                activation = "prelu",          # activation function
                alpha2::Int=2,                 # expansion factor in encoder stage 2
                alpha3::Int=3,                 # expansion factor in encoder stage 3
                edrops=(0.0, 0.0, 0.0),        # dropout rates for encoder
                ddrops=(0.0, 0.0),             # dropout rates for decoder
)
    # activations
    act_16     = activation == "prelu" ? PReLU(16) : activation
    act_ch_out = activation == "prelu" ? PReLU(ch_out) : activation

    # encoder
    e1  = Chain(ConvK3(ch_in, 16; stride=2),
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
    d2 = Chain(ConvTranspK2(ch_out, ch_out; stride=2),
               BatchNorm(ch_out),
               act_ch_out,
               Dropout(ddrops[2]),
    )
    d1 = Chain(ESPBlock1(2*ch_out, ch_out; activation=activation, stride=1),
               ConvTranspK2(ch_out, ch_out; stride=2),
               BatchNorm(ch_out),
               act_ch_out,
               Dropout(ddrops[1]),
    )
    d0 = Chain(ConvK1(2*ch_out, ch_out),
               ConvTranspK2(ch_out, ch_out; stride=2),   # no bn, no activation
    )

    # output chains
    encoder = Chain(e1=e1, e2a=e2a, e2b=e2b, e3a=e3a, e3b=e3b)
    bridges = Chain(b1=b1, b2=b2, b3=b3)
    decoder = Chain(d2=d2, d1=d1, d0=d0)

    return espnet(encoder, bridges, decoder)   # struct output
end


function (m::espnet)(x)
    # input image downsampling
    x1 = downsampling(x)
    x2 = downsampling(x1)

    # encoder
    out1 = m.encoder[:e1](x)
    ct1 = cat(x1, out1, dims=3)

    out2a = m.encoder[:e2a](ct1)
    out2b = m.encoder[:e2b](out2a)
    ct2 = cat(x2, out2a, out2b, dims=3)
    
    out3a = m.encoder[:e3a](ct2)
    out3b = m.encoder[:e3b](out3a)
    ct3 = cat(out3a, out3b, dims=3)


    # bridges
    b1 = m.bridges[:b1](ct1)
    b2 = m.bridges[:b2](ct2)
    b3 = m.bridges[:b3](ct3)


    # decoder
    d2 = m.decoder[:d2](b3)
    ct4 = cat(b2, d2, dims=3)

    d1 = m.decoder[:d1](ct4)
    ct5 = cat(b1, d1, dims=3)

    logits = m.decoder[:d0](ct5)

    feature_maps = [out1, ct1, out2a, out2b, ct2, out3a, out3b, ct3, # encoder [1:8]
                    b1, b2, b3,                                      # bridges [9:11]
                    d2, ct4, d1, ct5, logits]                        # decoder [12:16]
    return feature_maps
end


function ESPNet(ch_in::Int=3, ch_out::Int=1; activation="prelu")   # input/output channels
    model = espnet(ch_in, ch_out;
                   activation=activation,
                   alpha2=5,
                   alpha3=8,
                   edrops=(0.0, 0.1, 0.3),
                   ddrops=(0.0, 0.0),
    )
    act = ch_out == 1 ? x -> σ.(x) : x -> softmax(x; dims=3)
    return Chain(model, x->x[end], act)
end
