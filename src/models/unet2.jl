struct UNet2
    enc::Chain
    dec::Chain
    verbose::Bool
end


function UNet2(ch_in::Int=3, ch_out::Int=1;   # input/output channels
               activation    = relu,          # activation function
               alpha         = 1.0,           # feature channels multiplier
               verbose::Bool = false,         # logits output
)

    chs = alpha .* defaultChannels .|> Int

    # contracting path
    c1 = Chain(ConvK3(ch_in, chs[1], activation),
               Dropout(0.1),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation)
    )

    c2 = Chain(MaxPoolK2,
               ConvK3(chs[1], chs[2], activation),
               Dropout(0.2),
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation)
    )
    

    # expansive path
    e2 = Chain(ConvTranspK2(chs[2], chs[1]; stride=2), BatchNorm(chs[1], activation),
    )

    e1 = Chain(ConvK3(chs[2], chs[1], activation),
               Dropout(0.1),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
               ConvK1(chs[1], ch_out, identity)
    )

    e0 = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1=c1, c2=c2)
    dec = Chain(e2=e2, e1=e1, e0=e0)

    return UNet2(enc, dec, verbose)   # struct with encoder and decoder
end


function (m::UNet2)(x)
    enc1 = m.enc[:c1](x)
    enc2 = m.enc[:c2](enc1)

    dec2 = m.dec[:e2](enc2)
    dec1 = m.dec[:e1](cat(enc1, dec2; dims=3))

    yhat   = m.dec[:e0](dec1)
    logits = [enc1, enc2, dec2, dec1]

    if m.verbose   return yhat, logits   # logits output
    else           return yhat           # model output
    end
end

@functor UNet2
