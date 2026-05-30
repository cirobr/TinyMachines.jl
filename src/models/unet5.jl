struct unet5
    encoder::Chain
    upconvs::Chain
    decoder::Chain
end
@layer unet5

function unet5(ch_in::Int=3, ch_out::Int=2;          # input/output channels
               activation::Function = relu,          # activation function
               alpha::Int           = 1,             # channels divider
               edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
               ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)

    chs = defaultChannels .÷ alpha

    # encoder
    e1 = Chain(CB(ch_in, chs[1], activation),   Dropout(edrops[1]))
    e2 = Chain(MCB(chs[1], chs[2], activation), Dropout(edrops[2]))
    e3 = Chain(MCB(chs[2], chs[3], activation), Dropout(edrops[3]))
    e4 = Chain(MCB(chs[3], chs[4], activation), Dropout(edrops[4]))
    e5 = Chain(MCB(chs[4], chs[5], activation), Dropout(edrops[5]))

    # up convolutions
    u4 = ConvTranspK2(chs[5], chs[4], activation; stride=2)
    u3 = ConvTranspK2(chs[4], chs[3], activation; stride=2)
    u2 = ConvTranspK2(chs[3], chs[2], activation; stride=2)
    u1 = ConvTranspK2(chs[2], chs[1], activation; stride=2)

    # decoder
    d4 = Chain(CB(chs[5], chs[4], activation), Dropout(ddrops[4]))
    d3 = Chain(CB(chs[4], chs[3], activation), Dropout(ddrops[3]))
    d2 = Chain(CB(chs[3], chs[2], activation), Dropout(ddrops[2]))
    d1 = Chain(CB(chs[2], chs[1], activation), Dropout(ddrops[1]))
    
    d0 = ConvK1(chs[1], ch_out)

    # output chains
    encoder = Chain(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5)
    upconvs = Chain(u4=u4, u3=u3, u2=u2, u1=u1)
    decoder = Chain(d4=d4, d3=d3, d2=d2, d1=d1, d0=d0)

    return unet5(encoder, upconvs, decoder)   # struct output
end


function (m::unet5)(x::AbstractArray; return_features::Bool = false)
    enc1 = m.encoder.layers.e1(x)
    enc2 = m.encoder.layers.e2(enc1)
    enc3 = m.encoder.layers.e3(enc2)
    enc4 = m.encoder.layers.e4(enc3)
    enc5 = m.encoder.layers.e5(enc4)

    up4 = m.upconvs.layers.u4(enc5)
    cat4 = cat(enc4, up4; dims=3)
    dec4 = m.decoder.layers.d4(cat4)

    up3 = m.upconvs.layers.u3(dec4)
    cat3 = cat(enc3, up3; dims=3)
    dec3 = m.decoder.layers.d3(cat3)
    
    up2 = m.upconvs.layers.u2(dec3)
    cat2 = cat(enc2, up2; dims=3)
    dec2 = m.decoder.layers.d2(cat2)

    up1 = m.upconvs.layers.u1(dec2)
    cat1 = cat(enc1, up1; dims=3)
    dec1 = m.decoder.layers.d1(cat1)

    logits = m.decoder.layers.d0(dec1)

    # output features, logits
    if return_features
        return (logits  = logits,
                encoder = (enc1=enc1, enc2=enc2, enc3=enc3, enc4=enc4, enc5=enc5)
        )
    else
        return logits
    end
end
const unet = unet5


function UNet5(ch_in::Int=3, ch_out::Int=2;    # input/output channels
               activation::Function = relu,    # activation function
)
    return unet5(ch_in, ch_out;
                  activation=activation,
                  alpha=1,
                  edrops=(0.0, 0.0, 0.1, 0.2, 0.25),
                  ddrops=(0.0, 0.0, 0.1, 0.2),
    )
end
const UNet = UNet5
