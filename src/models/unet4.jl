struct UNet4
    encoder::Chain
    upconvs::Chain
    decoder::Chain
    verbose::Bool
end
@layer UNet4 trainable=(encoder, upconvs, decoder)

function UNet4(ch_in::Int=3, ch_out::Int=1;   # input/output channels
                activation    = relu,          # activation function
                alpha::Int    = 1,             # channels divider
                verbose::Bool = false,         # output feature maps
                dropc1::Float64 = 0.1,           # dropout rate c1
                dropc2::Float64 = 0.1,           # dropout rate c2
                dropc3::Float64 = 0.1,           # dropout rate c3
                dropc4::Float64 = 0.1,           # dropout rate c4
                drope3::Float64 = 0.1,           # dropout rate e3
                drope2::Float64 = 0.1,           # dropout rate e2
                drope1::Float64 = 0.1,           # dropout rate e1
)

    chs = defaultChannels .÷ alpha

    # contracting path
    c1 = CBlock(ch_in, chs[1], activation)
    c1 = Chain(c1, Dropout(dropc1))
    c2 = MCBlock(chs[1], chs[2], activation)
    c2 = Chain(c2, Dropout(dropc2))
    c3 = MCBlock(chs[2], chs[3], activation)
    c3 = Chain(c3, Dropout(dropc3))
    c4 = MCBlock(chs[3], chs[4], activation)
    c4 = Chain(c4, Dropout(dropc4))

    # up convolutions
    u3 = UpBlock(chs[4], chs[3], activation)
    u2 = UpBlock(chs[3], chs[2], activation)
    u1 = UpBlock(chs[2], chs[1], activation)

    # expansive path
    e3 = CBlock(chs[4], chs[3], activation)
    e3 = Chain(e3, Dropout(drope3))
    e2 = CBlock(chs[3], chs[2], activation)
    e2 = Chain(e2, Dropout(drope2))
    e1 = CBlock(chs[2], chs[1], activation)
    e1 = Chain(e1, Dropout(drope1))
    
    e0 = ConvK1(chs[1], ch_out)
    act = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    # output chains
    encoder = Chain(c1=c1, c2=c2, c3=c3, c4=c4)
    upconvs = Chain(u3=u3, u2=u2, u1=u1)
    decoder = Chain(e3=e3, e2=e2, e1=e1, e0=e0, act=act)

    return UNet4(encoder, upconvs, decoder, verbose)   # struct output
end


function (m::UNet4)(x::AbstractArray{Float32,4})
    enc1 = m.encoder[:c1](x)
    enc2 = m.encoder[:c2](enc1)
    enc3 = m.encoder[:c3](enc2)
    enc4 = m.encoder[:c4](enc3)

    
    up3 = m.upconvs[:u3](enc4)
    cat3 = cat(enc3, up3; dims=3)
    dec3 = m.decoder[:e3](cat3)
    
    up2 = m.upconvs[:u2](dec3)
    cat2 = cat(enc2, up2; dims=3)
    dec2 = m.decoder[:e2](cat2)

    up1 = m.upconvs[:u1](dec2)
    cat1 = cat(enc1, up1; dims=3)
    dec1 = m.decoder[:e1](cat1)

    dec0 = m.decoder[:e0](dec1)

    yhat         = m.decoder[:act](dec0)
    feature_maps = [enc1, enc2, enc3, enc4,   # encoder [1:4]
                    dec3, dec2, dec1, dec0]   # decoder [5:8]

    if m.verbose   return yhat, feature_maps   # feature maps output
    else           return yhat                 # model output
    end
end
