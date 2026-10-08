struct unet4
    encoder::Chain
    upconvs::Chain
    decoder::Chain
    eds::Chain
    dds::Chain
end
@layer unet4 trainable=(encoder,upconvs,decoder)

function unet4(
    ch_in::Int  = 3,                      # input channels
    ch_out::Int = 2;                      # output channels
    activation::Function = relu,          # activation function
    alpha::Int           = 1,             # channels divider
    edrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
    ddrops = (0.0, 0.0, 0.0),             # dropout rates
)
    # channels
    chs = defaultChannels .÷ alpha

    # dropouts
    ed1 = Dropout(edrops[1])
    ed2 = Dropout(edrops[2])
    ed3 = Dropout(edrops[3])
    ed4 = Dropout(edrops[4])

    dd1 = Dropout(ddrops[1])
    dd2 = Dropout(ddrops[2])
    dd3 = Dropout(ddrops[3])

    # encoder
    e1 = CB(ch_in, chs[1], activation)
    e2 = MCB(chs[1], chs[2], activation)
    e3 = MCB(chs[2], chs[3], activation)
    e4 = MCB(chs[3], chs[4], activation)

    # up convolutions
    u3 = ConvTrK2(chs[4], chs[3], activation)
    u2 = ConvTrK2(chs[3], chs[2], activation)
    u1 = ConvTrK2(chs[2], chs[1], activation)

    # decoder
    d3 = CB(chs[4], chs[3], activation)
    d2 = CB(chs[3], chs[2], activation)
    d1 = CB(chs[2], chs[1], activation)
    
    d0 = ConvK1(chs[1], ch_out)

    # output chains
    encoder = Chain(e1=e1, e2=e2, e3=e3, e4=e4)
    upconvs = Chain(u3=u3, u2=u2, u1=u1)
    decoder = Chain(d3=d3, d2=d2, d1=d1, d0=d0)
    eds = Chain(ed1=ed1, ed2=ed2, ed3=ed3, ed4=ed4)
    dds = Chain(dd1=dd1, dd2=dd2, dd3=dd3)

    return unet4(encoder, upconvs, decoder, eds, dds)   # struct output
end


function (m::unet4)(x::AbstractArray)
    # encoder
    enc1  = m.encoder.layers.e1(x)
    enc1d = m.eds.layers.ed1(enc1)
    enc2  = m.encoder.layers.e2(enc1d)
    enc2d = m.eds.layers.ed2(enc2)
    enc3  = m.encoder.layers.e3(enc2d)
    enc3d = m.eds.layers.ed3(enc3)
    enc4  = m.encoder.layers.e4(enc3d)
    enc4d = m.eds.layers.ed4(enc4)

    # decoder
    up3   = m.upconvs.layers.u3(enc4d)
    cat3  = cat(enc3, up3; dims=3)
    dec3  = m.decoder.layers.d3(cat3)
    dec3d = m.dds.layers.dd3(dec3)

    up2   = m.upconvs.layers.u2(dec3d)
    cat2  = cat(enc2, up2; dims=3)
    dec2  = m.decoder.layers.d2(cat2)
    dec2d = m.dds.layers.dd2(dec2)

    up1   = m.upconvs.layers.u1(dec2d)
    cat1  = cat(enc1, up1; dims=3)
    dec1  = m.decoder.layers.d1(cat1)
    dec1d = m.dds.layers.dd1(dec1)

    # logits
    return m.decoder.layers.d0(dec1d)
end


function UNet4(
    ch_in::Int  = 3,
    ch_out::Int = 2;
    activation::Function = relu,
)
    return unet4(
        ch_in,
        ch_out;
        activation=activation,
        alpha=1,
        edrops=(0.0, 0.0, 0.0, 0.5),
        ddrops=(0.0, 0.0, 0.0),
    )
end