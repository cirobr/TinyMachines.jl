struct UNet4
    enc::Chain
    dec::Chain
    verbose::Bool
end
@layer UNet4 trainable=(enc, dec)


function UNet4(ch_in::Int=3, ch_out::Int=1;   # input/output channels
               activation    = relu,          # activation function
               alpha         = 1.0,           # feature channels multiplier
               verbose::Bool = false,         # output feature maps
)

chs = alpha .* defaultChannels .|> Int

    # contracting path
    c1 = Chain(ConvK3(ch_in, chs[1]), activation,
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
            #    Dropout(0.1),
    )

    c2 = Chain(MaxPoolK2,
               ConvK3(chs[1], chs[2]), activation,
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation),
            #    Dropout(0.1),
    )
    
    c3 = Chain(MaxPoolK2,
               ConvK3(chs[2], chs[3]), activation,
               ConvK3(chs[3], chs[3]), BatchNorm(chs[3], activation),
               Dropout(0.1),
    )
    
    c4 = Chain(MaxPoolK2,
               ConvK3(chs[3], chs[4]), activation,
               ConvK3(chs[4], chs[4]), BatchNorm(chs[4], activation),
               Dropout(0.2),
    )
    

    # expansive path
    e4 = Chain(ConvTranspK2(chs[4], chs[3]; stride=2), activation,
    )

    e3 = Chain(ConvK3(chs[4], chs[3]), activation,
               ConvK3(chs[3], chs[3]), BatchNorm(chs[3], activation),
               Dropout(0.1),
               ConvTranspK2(chs[3], chs[2]; stride=2), activation,
    )
    
    e2 = Chain(ConvK3(chs[3], chs[2]), activation,
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation),
            #    Dropout(0.1),
               ConvTranspK2(chs[2], chs[1]; stride=2), activation,
    )
    
    e1 = Chain(ConvK3(chs[2], chs[1]), activation,
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
            #    Dropout(0.1),
    )
    
    e0 = ConvK1(chs[1], ch_out)
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1=c1, c2=c2, c3=c3, c4=c4)
    dec = Chain(e4=e4, e3=e3, e2=e2, e1=e1, e0=e0, act=act)

    return UNet4(enc, dec, verbose)   # struct with encoder and decoder
end


function (m::UNet4)(x)
    enc1 = m.enc[:c1](x)
    enc2 = m.enc[:c2](enc1)
    enc3 = m.enc[:c3](enc2)
    enc4 = m.enc[:c4](enc3)

    dec4 = m.dec[:e4](enc4)
    dec3 = m.dec[:e3](cat(enc3, dec4; dims=3))
    dec2 = m.dec[:e2](cat(enc2, dec3; dims=3))
    dec1 = m.dec[:e1](cat(enc1, dec2; dims=3))
    dec0 = m.dec[:e0](dec1)

    yhat         = m.dec[:act](dec0)
    feature_maps = [enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1, dec0]

    if m.verbose   return yhat, feature_maps   # feature maps output
    else           return yhat                 # model output
    end
end
