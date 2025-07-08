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



struct ArrayPReLU
    v::Array
end
@layer ArrayPReLU

function ArrayPReLU(ch::Int)
    v = rand32(1,1,ch)
    return ArrayPReLU(v)
end

function (m::ArrayPReLU)(x)
    return fpos.(x) .+ (m.v .* fneg.(x))
end