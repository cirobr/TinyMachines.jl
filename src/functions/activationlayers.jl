struct ConvPReLU
    conv::Conv
end
@layer ConvPReLU

function ConvPReLU(ch::Int)
    conv = DepthwiseConv((1, 1), ch => ch; # DepthwiseConv
        bias=false,
        init=rand32
    )
    return(ConvPReLU(conv))
end

function (m::ConvPReLU)(x)
    return fpos.(x) .+ (m.conv(fneg.(x)))
end
