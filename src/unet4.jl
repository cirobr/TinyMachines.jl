struct UNet4
    enc::Chain
    dec::Chain
    verbose::Bool
end


function UNet4(ch_in::Int=3, ch_out::Int=1;   # input/output channels
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
               Dropout(0.15),
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation)
    )
    
    c3 = Chain(MaxPoolK2,
               ConvK3(chs[2], chs[3], activation),
               Dropout(0.2),
               ConvK3(chs[3], chs[3]), BatchNorm(chs[3], activation)
    )
    
    c4 = Chain(MaxPoolK2,
               ConvK3(chs[3], chs[4], activation),
               Dropout(0.25),
               ConvK3(chs[4], chs[4]), BatchNorm(chs[4], activation)
    )
    

    # expansive path
    e4 = Chain(ConvTranspK2(chs[4], chs[3]), BatchNorm(chs[3], activation),
    )

    e3 = Chain(ConvK3(chs[4], chs[3], activation),
               Dropout(0.2),
               ConvK3(chs[3], chs[3], activation),
               ConvTranspK2(chs[3], chs[2]), BatchNorm(chs[2], activation)
    )
    
    e2 = Chain(ConvK3(chs[3], chs[2], activation),
               Dropout(0.15),
               ConvK3(chs[2], chs[2], activation),
               ConvTranspK2(chs[2], chs[1]), BatchNorm(chs[1], activation)
    )
    
    e1 = Chain(ConvK3(chs[2], chs[1], activation),
               Dropout(0.1),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
               ConvK1(chs[1], ch_out)
    )
    
    e0 = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1, c2, c3, c4)
    dec = Chain(e4, e3, e2, e1, e0)
    return UNet4(enc, dec, verbose)   # struct with encoder and decoder
end


function (m::UNet4)(x)
    enc1 = m.enc[1](x)
    enc2 = m.enc[2](enc1)
    enc3 = m.enc[3](enc2)
    enc4 = m.enc[4](enc3)

    dec4 = m.dec[1](enc4)
    dec3 = m.dec[2](cat(enc3, dec4; dims=3))
    dec2 = m.dec[3](cat(enc2, dec3; dims=3))
    dec1 = m.dec[4](cat(enc1, dec2; dims=3))

    yhat   = m.dec[end](dec1)
    logits = [enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1]

    if m.verbose   return yhat, logits   # logits output
    else           return yhat           # model output
    end
end

@functor UNet4
