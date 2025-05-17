# PReLU
DwConvPrelu(ch_in::Int, activation::Function=identity) =
    Chain(DepthwiseConv((1, 1), ch_in => ch_in, activation;
                        bias=false,
                        init=rand32)   ### TODO: use a better init (e.g. 0.1) ###
    )

struct ConvPReLU
    conv::Chain
end
@layer ConvPReLU

function ConvPReLU(ch_in::Int)
    return ConvPReLU(DwConvPrelu(ch_in, identity))
end

function (m::ConvPReLU)(x)
    return max.(x, 0) + m.conv(min.(x, 0))
end
