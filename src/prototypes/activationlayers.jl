### PReLU prototypes


# depthwise conv
ActivK1(ch_in::Int, activation::Function=Flux.leakyrelu) =
    Chain(DepthwiseConv((1, 1), ch_in => ch_in;
                        bias=false,
                        init=rand32),
          activation
)


# depthwise conv + PReLU
struct ConvPReLU
    conv::Flux.Conv
end
@layer ConvPReLU

function ConvPReLU(ch_in::Int)
    ActivK1(ch_in, identity)
end

function (m::ConvPReLU)(x)
    return max.(x, 0) .+ m.conv(min.(x, 0))
end


# PReLU layer   ### works in cpu, fails at gpucompiler.jl ###
struct PReLUlayer
    α::AbstractVector
    ch_in::Int
end
@layer PReLUlayer trainable=(α)

function PReLUlayer(ch_in::Int)
    α = rand(ch_in)
    return PReLUlayer(α, ch_in)
end

function (m::PReLUlayer)(x)
    α_broadcast = reshape(convert.(eltype(x), m.α), (1, 1, m.ch_in, 1))
    return max.(x, 0) .+ α_broadcast .* min.(x, 0)
end
