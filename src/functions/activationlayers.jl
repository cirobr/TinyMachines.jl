### PReLU prototypes


# depthwise conv
DwConvK1(ch_in::Int, activation::Function=identity) =
    Chain(DepthwiseConv((1, 1), ch_in => ch_in, activation;
                        bias=false,
                        init=rand32)
    )

# depthwise conv + PReLU
struct ConvPReLU
    conv::Chain
end
@layer ConvPReLU

function ConvPReLU(ch_in::Int)
    return ConvPReLU(DwConvK1(ch_in, identity))
end

function (m::ConvPReLU)(x)
    return max.(x, 0) + m.conv(min.(x, 0))
end


# PReLU layer
oftf(x, y) = oftype(float(x), y)

struct PReLUlayer
    a::AbstractArray
    # ch_in::Int
end
@layer PReLUlayer #trainable=(a)

function PReLUlayer(ch_in::Int)
    a = rand(1,1,ch_in,1)
    # a = rand(ch_in)

    return PReLUlayer(a)
end

function (m::PReLUlayer)(x)
    return @. max(x, 0) + oftf(x, m.a) * min(x, 0)
end
