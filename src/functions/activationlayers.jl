# PReLU
# preluweights(ch_in::Int) = DepthwiseConv((1, 1), ch_in => ch_in;
#                                  bias=false,
#                                  init=rand32
# )

struct ConvPReLU
    conv
end
@layer ConvPReLU

function ConvPReLU(ch_in::Int)
    # return ConvPReLU(preluweights(ch_in))
    conv = DepthwiseConv((1, 1), ch_in => ch_in;
        bias=false,
        init=rand32
    )
    return(ConvPReLU(conv))
end

function (m::ConvPReLU)(x)
    # return fpos.(float(x)) .+ (m.conv(fneg.(float(x))))
    return fpos.(x) .+ (m.conv(fneg.(x)))
end
