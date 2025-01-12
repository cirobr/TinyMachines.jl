struct UNet5
    enc #::Chain
    upc #::Chain
    dec #::Chain
    verbose::Bool
end
@layer UNet5 trainable=(enc, upc, dec)

function UNet5(ch_in::Int=3, ch_out::Int=1;   # input/output channels
               activation    = relu,          # activation function
               alpha::Int    = 1,             # channels divider
               verbose::Bool = false,         # output feature maps
)

    chs = defaultChannels .÷ alpha

    # contracting path
    c1 = cb(ch_in, chs[1], activation)
    c2 = mcb(chs[1], chs[2], activation)
    c3 = mcb(chs[2], chs[3], activation)
    c3 = Chain(c3, Dropout(0.1))
    c4 = mcb(chs[3], chs[4], activation)
    c4 = Chain(c4, Dropout(0.2))
    c5 = mcb(chs[4], chs[5], activation)
    c5 = Chain(c5, Dropout(0.25))

    # up convolutions
    u4 = ConvTranspK2(chs[5], chs[4], activation; stride=2)
    u3 = ConvTranspK2(chs[4], chs[3], activation; stride=2)
    u2 = ConvTranspK2(chs[3], chs[2], activation; stride=2)
    u1 = ConvTranspK2(chs[2], chs[1], activation; stride=2)

    # expansive path
    e4 = cb(chs[5], chs[4], activation)
    e4 = Chain(e4, Dropout(0.2))
    e3 = cb(chs[4], chs[3], activation)
    e3 = Chain(e3, Dropout(0.1))
    e2 = cb(chs[3], chs[2], activation)
    e1 = cb(chs[2], chs[1], activation)
    
    e0 = ConvK1(chs[1], ch_out)
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    enc = Chain(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)
    upc = Chain(u4=u4, u3=u3, u2=u2, u1=u1)
    dec = Chain(e4=e4, e3=e3, e2=e2, e1=e1, e0=e0, act=act)

    return UNet5(enc, upc, dec, verbose)   # struct arguments
end


function (m::UNet5)(x::AbstractArray{Float32,4})
    enc1 = m.enc[:c1](x)
    enc2 = m.enc[:c2](enc1)
    enc3 = m.enc[:c3](enc2)
    enc4 = m.enc[:c4](enc3)
    enc5 = m.enc[:c5](enc4)

    up4 = m.upc[:u4](enc5)
    cat4 = cat(enc4, up4; dims=3)   # cat: allocation source ???
    dec4 = m.dec[:e4](cat4)

    up3 = m.upc[:u3](dec4)
    cat3 = cat(enc3, up3; dims=3)
    dec3 = m.dec[:e3](cat3)
    
    up2 = m.upc[:u2](dec3)
    cat2 = cat(enc2, up2; dims=3)
    dec2 = m.dec[:e2](cat2)

    up1 = m.upc[:u1](dec2)
    cat1 = cat(enc1, up1; dims=3)
    dec1 = m.dec[:e1](cat1)

    dec0 = m.dec[:e0](dec1)

    yhat         = m.dec[:act](dec0)
    feature_maps = [enc1, enc2, enc3, enc4, enc5,   # encoder [1:5]
                    dec4, dec3, dec2, dec1, dec0]   # decoder [6:10]

    if m.verbose   return yhat, feature_maps   # feature maps output
    else           return yhat                 # model output
    end
end
