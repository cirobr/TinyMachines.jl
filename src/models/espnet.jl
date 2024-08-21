struct ESPnet
    downsampling
    encoder::Chain
    bridge::Chain
    decoder::Chain
    verbose::Bool
end
@layer ESPnet


function ESPnet(ch_in::Int, ch_out::Int;
                activation=relu,
                alpha2::Int=2, alpha3::Int=3,
                verbose::Bool=false,
)
    # downsampling
    ds = Flux.MeanPool((3,3); pad=SamePad(), stride=2)


    # encoder
    e1  = Chain(ConvK3(ch_in, 16; stride=2), BatchNorm(16, activation))

    e2a = ESPModule1(19, 64, activation; stride=2, add=false)
    # e2a = Chain(e2a, Dropout(0.0))
    
    e2b = ESPModule4_alpha(64, activation; alpha=alpha2)
    e2b = Chain(e2b, Dropout(0.1))
    
    e3a = ESPModule1(131, 128, activation; stride=2, add=false)
    
    e3b = ESPModule4_alpha(128, activation; alpha=alpha3)
    e3b = Chain(e3b, Dropout(0.3))


    # bridges
    b1 = ConvK1(19,  ch_out)
    b2 = ConvK1(131, ch_out)
    b3 = ConvK1(256, ch_out)


    # decoder
    d3 = Chain(ConvTranspK2(ch_out, ch_out; stride=2), BatchNorm(ch_out, activation))

    d2 = Chain(ESPModule1(2*ch_out, ch_out, activation; stride=1, add=false),
               ConvTranspK2(ch_out, ch_out; stride=2), BatchNorm(ch_out, activation)
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

    return ESPnet(downsampling, encoder, bridge, decoder, verbose)
end


function (m::ESPnet)(x)
    # encoder
    ds1 = m.downsampling(x);              # @show size(ds1)
    out1 = m.encoder[:e1](x);             # @show size(out1)
    ct1 = cat(ds1, out1, dims=3);         # @show size(ct1)

    ds2 = m.downsampling(ds1);            # @show size(ds2)
    out2a = m.encoder[:e2a](ct1);         # @show size(out2a)
    out2b = m.encoder[:e2b](out2a);       # @show size(out2b)
    ct2 = cat(ds2, out2a, out2b, dims=3); # @show size(ct2)
    
    out3a = m.encoder[:e3a](ct2);         # @show size(out3a)
    out3b = m.encoder[:e3b](out3a);       # @show size(out3b)
    ct3 = cat(out3a, out3b, dims=3);      # @show size(ct3)


    # bridges
    b1 = m.bridge[:b1](ct1);              # @show size(b1)
    b2 = m.bridge[:b2](ct2);              # @show size(b2)
    b3 = m.bridge[:b3](ct3);              # @show size(b3)


    # decoder
    d3 = m.decoder[:d3](b3);              # @show size(d3)
    ct4 = cat(b2, d3, dims=3);            # @show size(ct4)

    d2 = m.decoder[:d2](ct4);             # @show size(d2)
    ct5 = cat(b1, d2, dims=3);            # @show size(ct5)

    d1 = m.decoder[:d1](ct5);             # @show size(d1)
    yhat = m.decoder[:d0](d1);            # @show size(yhat)


    # return yhat
    if m.verbose   return yhat, d1   # model and logits outputs
    else           return yhat       # model output
    end
end
