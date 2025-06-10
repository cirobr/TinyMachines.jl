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

# function (m::ConvPReLU)(x)
#     xmax = max.(x, 0)
#     xmin = min.(x, 0)
#     xconv = m.conv(xmin)
#     # @assert size(xmax) == size(xconv) "Size mismatch: $(size(xmax)) != $(size(xconv))"
#     return xmax + xconv
# end
