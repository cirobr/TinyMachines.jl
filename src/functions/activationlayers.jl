# PReLU
preluweights(ch_in::Int) = DepthwiseConv((1, 1), ch_in => ch_in;
                                 bias=false,
                                 init=rand32
)

struct ConvPReLU
    conv
end
@layer ConvPReLU

function ConvPReLU(ch_in::Int)
    return ConvPReLU(preluweights(ch_in))
end

function (m::ConvPReLU)(x)
    return max.(x, 0) .+ m.conv(min.(x, 0))
end