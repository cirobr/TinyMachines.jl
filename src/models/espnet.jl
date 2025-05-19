struct ESPNet
    downsampling
    encoder::Chain
    bridge::Chain
    decoder::Chain
    verbose::Bool
end
@layer ESPNet trainable=(encoder, bridge, decoder)


function ESPNet(ch_in::Int=3, ch_out::Int=1;
                # activation::Function=relu,   # replaced by ConvPReLU
                alpha2::Int=2, alpha3::Int=3,
                verbose::Bool=false,
)
    # downsampling
    ds = Flux.MeanPool((3,3); pad=SamePad(), stride=2)


    # encoder
    e1  = Chain(ConvK3(ch_in, 16; stride=2),
                BatchNorm(16),
                ConvPReLU(16)
    )

    e2a = ESPBlock1(19, 64; stride=2, add=false)
    # e2a = Chain(e2a, Dropout(0.0))
    
    e2b = ESPBlock4_alpha(64; alpha=alpha2)
    e2b = Chain(e2b, Dropout(0.1))
    
    e3a = ESPBlock1(131, 128; stride=2, add=false)
    
    e3b = ESPBlock4_alpha(128; alpha=alpha3)
    e3b = Chain(e3b, Dropout(0.3))


    # bridges
    b1 = ConvK1(19,  ch_out)
    b2 = ConvK1(131, ch_out)
    b3 = ConvK1(256, ch_out)


    # decoder
    d3 = Chain(ConvTranspK2(ch_out, ch_out; stride=2),
               BatchNorm(ch_out),
               ConvPReLU(ch_out),
    )
    d2 = Chain(ESPBlock1(2*ch_out, ch_out; stride=1, add=false),
               ConvTranspK2(ch_out, ch_out; stride=2),
               BatchNorm(ch_out),
               ConvPReLU(ch_out),
    )
    d1 = Chain(ConvK1(2*ch_out, ch_out),
               ConvTranspK2(ch_out, ch_out; stride=2)   # no bn, no activation
    )
    d0 = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)


    # output chains
    downsampling = ds
    encoder = Chain(e1=e1, e2a=e2a, e2b=e2b, e3a=e3a, e3b=e3b)
    bridge  = Chain(b1=b1, b2=b2, b3=b3)
    decoder = Chain(d3=d3, d2=d2, d1=d1, d0=d0)

    return ESPNet(downsampling, encoder, bridge, decoder, verbose)
end


function (m::ESPNet)(x)
    # encoder
    ds1 = m.downsampling(x)
    out1 = m.encoder[:e1](x)
    ct1 = cat(ds1, out1, dims=3)

    ds2 = m.downsampling(ds1)
    out2a = m.encoder[:e2a](ct1)
    out2b = m.encoder[:e2b](out2a)
    ct2 = cat(ds2, out2a, out2b, dims=3)
    
    out3a = m.encoder[:e3a](ct2)
    out3b = m.encoder[:e3b](out3a)
    ct3 = cat(out3a, out3b, dims=3)


    # bridges
    b1 = m.bridge[:b1](ct1)
    b2 = m.bridge[:b2](ct2)
    b3 = m.bridge[:b3](ct3)


    # decoder
    d3 = m.decoder[:d3](b3)
    ct4 = cat(b2, d3, dims=3)

    d2 = m.decoder[:d2](ct4)
    ct5 = cat(b1, d2, dims=3)

    d1 = m.decoder[:d1](ct5)
    yhat = m.decoder[:d0](d1)


    feature_maps = [ds1, out1, ct1, ds2, out2a, out2b, ct2, out3a, out3b, ct3, # encoder [1:10]
                    b1, b2, b3,                                                # bridge  [11:13]
                    d3, ct4, d2, ct5, d1]                                      # decoder [14:18]

    # return yhat
    if m.verbose   return yhat, feature_maps   # model and logits outputs
    else           return yhat                 # model output
    end
end
