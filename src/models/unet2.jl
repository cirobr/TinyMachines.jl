struct UNet2
    enc::Chain
    upc::Chain
    dec::Chain
    verbose::Bool
end
@layer UNet2 trainable=(enc, upc, dec)


function UNet2(ch_in::Int=3, ch_out::Int=1;   # input/output channels
               activation    = relu,          # activation function
               alpha::Int    = 1,             # feature channels divider
               verbose::Bool = false,         # output feature maps
)

    chs = defaultChannels .÷ alpha

    # contracting path
    c1 = Chain(ConvK3(ch_in, chs[1], activation),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
               Dropout(0.1),
    )

    c2 = Chain(MaxPoolK2,
               ConvK3(chs[1], chs[2], activation),
               ConvK3(chs[2], chs[2]), BatchNorm(chs[2], activation),
               Dropout(0.2),
    )
    

    # up convolutions
    upc = Chain(ConvTranspK2(chs[2], chs[1], activation; stride=2),
    )


    # expansive path
    e1 = Chain(ConvK3(chs[2], chs[1], activation),
               ConvK3(chs[1], chs[1]), BatchNorm(chs[1], activation),
               Dropout(0.1),
    )
    
    e0 = ConvK1(chs[1], ch_out)
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1, c2)
    dec = Chain(e1, e0, act)

    return UNet2(enc, upc, dec, verbose)   # struct arguments
end


function (m::UNet2)(x)
    enc1 = m.enc[1](x)
    enc2 = m.enc[2](enc1)

    up1 = m.upc(enc2)
    dec1 = m.dec[1](cat(enc1, up1; dims=3))
    dec0 = m.dec[2](dec1)

    yhat         = m.dec[end](dec0)
    feature_maps = [enc1, enc2, dec1, dec0]

    if m.verbose   return yhat, feature_maps   # feature maps output
    else           return yhat                 # model output
    end
end
