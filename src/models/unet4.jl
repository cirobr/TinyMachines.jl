struct unet4
    encoder::Chain
    upconvs::Chain
    decoder::Chain
end
@layer unet4


function unet4(ch_in::Int=3, ch_out::Int=1;    # input/output channels
               activation::Function = relu,    # activation function
               alpha::Int           = 1,       # channels divider
               edrops = (0.0, 0.0, 0.0, 0.0),  # dropout rates
               ddrops = (0.0, 0.0, 0.0),       # dropout rates
)

    chs = defaultChannels .÷ alpha

    # encoder
    e1 = Chain(CBlock(ch_in, chs[1], activation),   Dropout(edrops[1]))
    e2 = Chain(MCBlock(chs[1], chs[2], activation), Dropout(edrops[2]))
    e3 = Chain(MCBlock(chs[2], chs[3], activation), Dropout(edrops[3]))
    e4 = Chain(MCBlock(chs[3], chs[4], activation), Dropout(edrops[4]))

    # up convolutions
    u3 = UpBlock(chs[4], chs[3], activation)
    u2 = UpBlock(chs[3], chs[2], activation)
    u1 = UpBlock(chs[2], chs[1], activation)

    # decoder
    d3 = Chain(CBlock(chs[4], chs[3], activation), Dropout(ddrops[3]))
    d2 = Chain(CBlock(chs[3], chs[2], activation), Dropout(ddrops[2]))
    d1 = Chain(CBlock(chs[2], chs[1], activation), Dropout(ddrops[1]))
    
    d0 = ConvK1(chs[1], ch_out)

    # output chains
    encoder = Chain(e1=e1, e2=e2, e3=e3, e4=e4)
    upconvs = Chain(u3=u3, u2=u2, u1=u1)
    decoder = Chain(d3=d3, d2=d2, d1=d1, d0=d0)

    return unet4(encoder, upconvs, decoder)   # struct output
end


function (m::unet4)(x::AbstractArray{Float32,4})
    enc1 = m.encoder[:e1](x)
    enc2 = m.encoder[:e2](enc1)
    enc3 = m.encoder[:e3](enc2)
    enc4 = m.encoder[:e4](enc3)

    
    up3 = m.upconvs[:u3](enc4)
    cat3 = cat(enc3, up3; dims=3)
    dec3 = m.decoder[:d3](cat3)
    
    up2 = m.upconvs[:u2](dec3)
    cat2 = cat(enc2, up2; dims=3)
    dec2 = m.decoder[:d2](cat2)

    up1 = m.upconvs[:u1](dec2)
    cat1 = cat(enc1, up1; dims=3)
    dec1 = m.decoder[:d1](cat1)

    dec0 = m.decoder[:d0](dec1)
    logits = dec0

    feature_maps = [enc1, enc2, enc3, enc4,     # encoder[1:4]
                    dec3, dec2, dec1, logits]   # decoder[5:8]
    return feature_maps
end


function UNet4(ch_in::Int=3, ch_out::Int=1;    # input/output channels
               activation::Function = relu,    # activation function
)
    model = unet4(ch_in, ch_out;
                  activation=activation,
                  alpha=1,
                  edrops=(0.0, 0.0, 0.1, 0.2),
                  ddrops=(0.0, 0.0, 0.1),
    )
    act = ch_out == 1 ? x -> σ.(x) : x -> softmax(x; dims=3)
    return Chain(model, x->x[end], act)
end
