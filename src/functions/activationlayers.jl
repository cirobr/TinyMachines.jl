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
    vector::AbstractArray
    # vector::Array{Float32, 3}
end
@layer ArrayPReLU

function ArrayPReLU(ch::Int)
    vector = rand32(1,1,ch)
    return ArrayPReLU(vector)
end

function (m::ArrayPReLU)(x)
    return fpos.(x) .+ (m.vector .* fneg.(x))
end



### working prelu
PReLU = ArrayPReLU
