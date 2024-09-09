struct UNet5
    enc::Chain
    upc::Chain
    dec::Chain
    verbose::Bool
end
@layer UNet5 trainable=(enc, upc, dec)


function UNet5(ch_in::Int=3, ch_out::Int=1;   # input/output channels
               activation    = relu,          # activation function
               alpha::Int    = 1,             # feature channels divider
               verbose::Bool = false,         # output feature maps
)

    chs = defaultChannels .÷ alpha

    # contracting path
    c1 = Chain(ConvK3(ch_in, chs[1], activation),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
            #    Dropout(0.1),
    )

    c2 = Chain(MaxPoolK2,
               ConvK3(chs[1], chs[2], activation),
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation),
            #    Dropout(0.1),
    )
    
    c3 = Chain(MaxPoolK2,
               ConvK3(chs[2], chs[3], activation),
               ConvK3(chs[3], chs[3]), BatchNorm(chs[3], activation),
               Dropout(0.1),
    )
    
    c4 = Chain(MaxPoolK2,
               ConvK3(chs[3], chs[4], activation),
               ConvK3(chs[4], chs[4]), BatchNorm(chs[4], activation),
               Dropout(0.2),
    )
    
    c5 = Chain(MaxPoolK2,
               ConvK3(chs[4], chs[5], activation),
               ConvK3(chs[5], chs[5]), BatchNorm(chs[5], activation),
               Dropout(0.25),
    )


    # up convolutions
    upc = Chain(ConvTranspK2(chs[5], chs[4], activation; stride=2),
                ConvTranspK2(chs[4], chs[3], activation; stride=2),
                ConvTranspK2(chs[3], chs[2], activation; stride=2),
                ConvTranspK2(chs[2], chs[1], activation; stride=2),
    )


    # expansive path
    e4 = Chain(ConvK3(chs[5], chs[4], activation),
               ConvK3(chs[4], chs[4]), BatchNorm(chs[4], activation),
               Dropout(0.2),
               
    )
    
    e3 = Chain(ConvK3(chs[4], chs[3], activation),
               ConvK3(chs[3], chs[3]), BatchNorm(chs[3], activation),
               Dropout(0.1),
    
    )
    
    e2 = Chain(ConvK3(chs[3], chs[2], activation),
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation),
            #    Dropout(0.1),
    )
    
    e1 = Chain(ConvK3(chs[2], chs[1], activation),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
            #    Dropout(0.1),
    )
    
    e0 = ConvK1(chs[1], ch_out)
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1, c2, c3, c4, c5)
    dec = Chain(e4, e3, e2, e1, e0, act)

    return UNet5(enc, upc, dec, verbose)   # struct arguments
end


function (m::UNet5)(x)
    enc1 = m.enc[1](x)
    enc2 = m.enc[2](enc1)
    enc3 = m.enc[3](enc2)
    enc4 = m.enc[4](enc3)
    enc5 = m.enc[5](enc4)

    up4 = m.upc[1](enc5)
    dec4 = m.dec[1](cat(enc4, up4; dims=3))
    up3 = m.upc[2](dec4)
    dec3 = m.dec[2](cat(enc3, up3; dims=3))
    up2 = m.upc[3](dec3)
    dec2 = m.dec[3](cat(enc2, up2; dims=3))
    up1 = m.upc[4](dec2)
    dec1 = m.dec[4](cat(enc1, up1; dims=3))
    dec0 = m.dec[5](dec1)

    yhat         = m.dec[end](dec0)
    feature_maps = [enc1, enc2, enc3, enc4, enc5, dec4, dec3, dec2, dec1, dec0]

    if m.verbose   return yhat, feature_maps   # feature maps output
    else           return yhat                 # model output
    end
end
