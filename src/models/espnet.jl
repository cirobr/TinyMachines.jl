# encoder stage 1
struct ESPEncoderStage1
    ds1::MeanPool
    e1::Chain
    fusion1::Chain
    edrop1::Dropout
end
@layer ESPEncoderStage1 trainable=(ds1, e1, fusion1)

function ESPEncoderStage1(ch_in::Int = 3; activation = "prelu", edrop = 0.0)
    act_16 = activation == "prelu" ? PReLU(16) : activation
    act_19 = activation == "prelu" ? PReLU(19) : activation

    ds1 = MeanPool((3, 3); stride=2, pad=1)
    e1  = Chain(ConvK3(ch_in, 16; stride=2), BatchNorm(16), act_16)
    fusion1 = Chain(BatchNorm(19), act_19)
    edrop1 = Dropout(edrop)

    return ESPEncoderStage1(ds1, e1, fusion1, edrop1)
end

function (m::ESPEncoderStage1)(x)
    e1_out   = m.e1(x)
    inp1     = m.ds1(x)
    e1_fused = m.fusion1(cat(e1_out, inp1; dims=3))
    e1_drop  = m.edrop1(e1_fused)          # continuing-path dropout
    return e1_drop, x                      # pass original image forward
end


# encoder stage 2
struct ESPEncoderStage2
    ds2::Chain
    e2_down::DownSampler
    e2_deep::Chain
    fusion2::Chain
    edrop2::Dropout
end
@layer ESPEncoderStage2 trainable=(ds2, e2_down, e2_deep, fusion2)

function ESPEncoderStage2(; activation = "prelu", alpha2::Int = 2, edrop = 0.0)
    act_131 = activation == "prelu" ? PReLU(131) : activation

    ds2 = Chain(
        MeanPool((3, 3); stride=2, pad=1),
        MeanPool((3, 3); stride=2, pad=1)
    )
    e2_down = DownSampler(19, 64; activation=activation)
    e2_deep = Chain([ESPModule(64, 64; add=true, activation=activation)
                     for _ in 1:alpha2]...)
    fusion2 = Chain(BatchNorm(131), act_131)
    edrop2 = Dropout(edrop)

    return ESPEncoderStage2(ds2, e2_down, e2_deep, fusion2, edrop2)
end

function (m::ESPEncoderStage2)((e1_drop, x))
    e2_down_out = m.e2_down(e1_drop)
    e2_deep_out = m.e2_deep(e2_down_out)
    inp2        = m.ds2(x)
    e2_fused    = m.fusion2(cat(e2_down_out, e2_deep_out, inp2; dims=3))
    e2_drop     = m.edrop2(e2_fused)
    return e2_drop, x
end


# encoder stage 3
struct ESPEncoderStage3
    e3_down::DownSampler
    e3_deep::Chain
    fusion3::Chain
    edrop3::Dropout
end
@layer ESPEncoderStage3 trainable=(e3_down, e3_deep, fusion3)

function ESPEncoderStage3(; activation = "prelu", alpha3::Int = 3, edrop = 0.0)
    act_256 = activation == "prelu" ? PReLU(256) : activation

    e3_down = DownSampler(131, 128; activation=activation)
    e3_deep = Chain([ESPModule(128, 128; add=true, activation=activation)
                     for _ in 1:alpha3]...)
    fusion3 = Chain(BatchNorm(256), act_256)
    edrop3 = Dropout(edrop)

    return ESPEncoderStage3(e3_down, e3_deep, fusion3, edrop3)
end

function (m::ESPEncoderStage3)((e2_drop, x))
    e3_down_out = m.e3_down(e2_drop)
    e3_deep_out = m.e3_deep(e3_down_out)
    e3_fused    = m.fusion3(cat(e3_down_out, e3_deep_out; dims=3))
    e3_drop     = m.edrop3(e3_fused)
    return e3_drop                         # final stage returns only the feature
end


# ─────────────────────────────────────────────────────────────
# Encoder = Chain of the three stages
# ─────────────────────────────────────────────────────────────
function ESPNetEncoder(
    ch_in::Int = 3;
    activation = "prelu",
    alpha2::Int = 2,
    alpha3::Int = 3,
    edrops = (0.0, 0.0, 0.0),
)
    e1 = ESPEncoderStage1(ch_in; activation, edrop=edrops[1])
    e2 = ESPEncoderStage2(; activation, alpha2, edrop=edrops[2])
    e3 = ESPEncoderStage3(; activation, alpha3, edrop=edrops[3])

    return Chain(e1=e1, e2=e2, e3=e3)
end


# ─────────────────────────────────────────────────────────────
# Full model (encoder is now the Chain above)
# ─────────────────────────────────────────────────────────────
struct espnet
    encoder::Chain          # Chain(stage1, stage2, stage3)
    bridge1::Conv
    bridge2::Conv
    bridge3::Conv
    d3_up::ConvTranspose
    d2_deep::Chain
    d2_up::Chain
    d1_conv::Chain
    d1_up::ConvTranspose
    dds::Chain
end
@layer espnet trainable=(
    encoder,
    bridge1, bridge2, bridge3,
    d3_up, d2_deep, d2_up, d1_conv, d1_up
)

function espnet(
    ch_in::Int = 3,
    ch_out::Int = 2;
    activation = "prelu",
    alpha2::Int = 2,
    alpha3::Int = 3,
    edrops = (0.0, 0.0, 0.0),
    ddrops = (0.0, 0.0),
)
    encoder = ESPNetEncoder(ch_in; activation, alpha2, alpha3, edrops)

    act_ch_out = activation == "prelu" ? PReLU(ch_out)     : activation
    act_2c     = activation == "prelu" ? PReLU(2 * ch_out) : activation

    bridge1 = ConvK1(19,  ch_out)
    bridge2 = ConvK1(131, ch_out)
    bridge3 = ConvK1(256, ch_out)

    d3_up = ConvTrK2(ch_out, ch_out)

    d2_deep = if ch_out < 5
        Chain(
            BatchNorm(2 * ch_out), act_2c,
            ConvK1(2 * ch_out, ch_out),
            ConvK3(ch_out, ch_out),
            BatchNorm(ch_out), act_ch_out,
        )
    else
        Chain(
            BatchNorm(2 * ch_out), act_2c,
            ESPModule(2 * ch_out, ch_out; add=false, activation=activation),
        )
    end

    d2_up = Chain(ConvTrK2(ch_out, ch_out), BatchNorm(ch_out), act_ch_out)
    d1_conv = Chain(ConvK1(2 * ch_out, ch_out), BatchNorm(ch_out), act_ch_out)
    d1_up = ConvTrK2(ch_out, ch_out)

    dds = Chain(
        ddrop2 = Dropout(ddrops[2]),
        ddrop1 = Dropout(ddrops[1]),
    )

    return espnet(encoder, bridge1, bridge2, bridge3,
                  d3_up, d2_deep, d2_up, d1_conv, d1_up, dds)
end

function (m::espnet)(x)
    # Run once and capture all intermediate fused maps
    acts = Flux.activations(m.encoder, x)

    e1_fused = acts[1][1]   # stage1 returns (feature, x)
    e2_fused = acts[2][1]   # stage2 returns (feature, x)
    e3_fused = acts[3]      # stage3 returns the pure feature

    b3 = m.bridge3(e3_fused)
    b2 = m.bridge2(e2_fused)
    b1 = m.bridge1(e1_fused)

    d3 = m.d3_up(b3)
    d2_cat = m.dds.layers.ddrop2(cat(b2, d3; dims=3))
    d2 = m.d2_up(m.d2_deep(d2_cat))
    d1_cat = m.dds.layers.ddrop1(cat(b1, d2; dims=3))
    return m.d1_up(m.d1_conv(d1_cat))
end

# paper-faithful:    edrops=(0.0, 0.0, 0.0), ddrops=(0.0, 0.0)
# mild overfitting:  edrops=(0.01, 0.05, 0.1), ddrops=(0.05, 0.1)
# clear overfitting: edrops=(0.05, 0.1, 0.2), ddrops=(0.1, 0.15)
function ESPNet(
    ch_in::Int=3,
    ch_out::Int=2;
    activation="prelu"
)
    return espnet(
        ch_in,
        ch_out;
        activation=activation,
        alpha2=2,          # matches frequently cited paper value (α2=2)
        alpha3=8,          # matches frequently cited paper value (α3=8)
        edrops=(0.0, 0.0, 0.0),
        ddrops=(0.0, 0.0),
    )
end
