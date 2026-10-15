struct unet5H
    encoder::Chain
    upconvs::Chain
    decoder::Chain
    eds::Chain
    dds::Chain
end
@layer unet5H trainable=(encoder,upconvs,decoder)

function unet5H(
    ch_in::Int  = 3,                      # input channels
    ch_out::Int = 2;                      # output channels
    activation::Function = relu,          # activation function
    alpha::Int           = 1,             # channels divider factor
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
    ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)
    # channels
    chs = defaultChannels .÷ alpha

    # dropouts
    ed1 = Dropout(edrops[1])
    ed2 = Dropout(edrops[2])
    ed3 = Dropout(edrops[3])
    ed4 = Dropout(edrops[4])
    ed5 = Dropout(edrops[5])

    dd1 = Dropout(ddrops[1])
    dd2 = Dropout(ddrops[2])
    dd3 = Dropout(ddrops[3])
    dd4 = Dropout(ddrops[4])

    # encoder
    e1 = CB(ch_in, chs[1], activation)
    e2 = CB(chs[1], chs[2], activation)
    e3 = CB(chs[2], chs[3], activation)
    e4 = CB(chs[3], chs[4], activation)
    e5 = CB(chs[4], chs[5], activation)

    # up convolutions
    u4 = ConvTrK2(chs[5], chs[4], activation)
    u3 = ConvTrK2(chs[4], chs[3], activation)
    u2 = ConvTrK2(chs[3], chs[2], activation)
    u1 = ConvTrK2(chs[2], chs[1], activation)

    # decoder
    d4 = CB(chs[5]+chs[3]*3, chs[4], activation)
    d3 = CB(chs[4]+chs[2]*3, chs[3], activation)
    d2 = CB(chs[3]+chs[1]*3, chs[2], activation)
    d1 = CB(chs[2], chs[1], activation)
    
    d0 = ConvK1(chs[1], ch_out)

    # output chains
    encoder = Chain(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5)
    upconvs = Chain(u4=u4, u3=u3, u2=u2, u1=u1)
    decoder = Chain(d4=d4, d3=d3, d2=d2, d1=d1, d0=d0)
    eds = Chain(ed1=ed1, ed2=ed2, ed3=ed3, ed4=ed4, ed5=ed5)
    dds = Chain(dd1=dd1, dd2=dd2, dd3=dd3, dd4=dd4)

    return unet5H(encoder, upconvs, decoder, eds, dds)   # struct output
end


function (m::unet5H)(x::AbstractArray)
    # encoder
    enc1  = m.encoder.layers.e1(x)
    enc1d = m.eds.layers.ed1(enc1)

    ll2, lh2, hl2, hh2 = haar_down(enc1d)#; @show size(ll2)
    enc2  = m.encoder.layers.e2(ll2)
    enc2d = m.eds.layers.ed2(enc2)
    
    ll3, lh3, hl3, hh3 = haar_down(enc2d)#; @show size(ll3)
    enc3  = m.encoder.layers.e3(ll3)
    enc3d = m.eds.layers.ed3(enc3)
    
    ll4, lh4, hl4, hh4 = haar_down(enc3d)#; @show size(ll4)
    enc4  = m.encoder.layers.e4(ll4)
    enc4d = m.eds.layers.ed4(enc4)
    
    ll5, _, _, _ = haar_down(enc4d)
    enc5  = m.encoder.layers.e5(ll5)
    enc5d = m.eds.layers.ed5(enc5)

    # decoder
    up4   = m.upconvs.layers.u4(enc5d)
    # cat4  = cat(enc4, up4; dims=3)
    cat4  = cat(enc4, up4, lh4, hl4, hh4; dims=3)
    dec4  = m.decoder.layers.d4(cat4)
    dec4d = m.dds.layers.dd4(dec4)

    up3   = m.upconvs.layers.u3(dec4d)
    # cat3  = cat(enc3, up3; dims=3)
    cat3  = cat(enc3, up3, lh3, hl3, hh3; dims=3)
    dec3  = m.decoder.layers.d3(cat3)
    dec3d = m.dds.layers.dd3(dec3)

    up2   = m.upconvs.layers.u2(dec3d)
    # cat2  = cat(enc2, up2; dims=3)
    cat2  = cat(enc2, up2, lh2, hl2, hh2; dims=3)
    dec2  = m.decoder.layers.d2(cat2)
    dec2d = m.dds.layers.dd2(dec2)

    up1   = m.upconvs.layers.u1(dec2d)
    cat1  = cat(enc1, up1; dims=3)
    dec1  = m.decoder.layers.d1(cat1)
    dec1d = m.dds.layers.dd1(dec1)

    # logits
    return m.decoder.layers.d0(dec1d)
end


# paper-faithful:    edrops=(0.0, 0.0, 0.0, 0.0, 0.5), ddrops=(0.0, 0.0, 0.0, 0.0)
# mild overfitting:  edrops=(0.0, 0.0, 0.10, 0.20, 0.25), ddrops=(0.0, 0.0, 0.10, 0.20)
# clear overfitting: edrops=(0.05, 0.10, 0.15, 0.20, 0.30), ddrops=(0.00, 0.05, 0.10, 0.15)
function UNet5H(
    ch_in::Int  = 3,
    ch_out::Int = 2;
    activation::Function = relu,
)
    return unet5H(
        ch_in,
        ch_out;
        activation = activation,
        alpha = 1,
        edrops = (0.0, 0.0, 0.0, 0.0, 0.5),
        ddrops = (0.0, 0.0, 0.0, 0.0),
    )
end