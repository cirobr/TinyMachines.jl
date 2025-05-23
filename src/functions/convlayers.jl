function ConvK1(ch_in::Int, ch_out::Int, activation::Function=identity)
    gn = kf * √(w1 * ch_in)

    return Conv((1,1), ch_in => ch_out, activation;
                bias=false,
                init=kaiming_normal(gain=gn)
    )
end
PointwiseConv = ConvK1


function ConvK2(ch_in::Int, ch_out::Int, activation::Function=identity)
    gn = kf * √(w2 * ch_in)

    return Conv((2,2), ch_in => ch_out, activation;
                stride=2,
                pad=SamePad(),
                bias=true,
                init=kaiming_normal(gain=gn)
    )
end


function ConvK3(ch_in::Int, ch_out::Int, activation::Function=identity;
                stride::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")
    gn = kf * √(w3 * ch_in)
    
    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                bias=true,
                init=kaiming_normal(gain=gn)
    )
end


function UpConvK2(ch_in::Int, ch_out::Int, activation::Function=identity)
    return Chain(Upsample(scale=(4,4)), ConvK2(ch_in, ch_out, activation))
end


function ConvTranspK2(ch_in::Int, ch_out::Int, activation::Function=identity;
                      stride::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")
    gn = kf * √(w2 * ch_in)
    
    return ConvTranspose((2,2), ch_in => ch_out, activation;
                         stride=stride,
                         pad=SamePad(),
                         bias=true,
                         init=kaiming_normal(gain=gn)
    )
end


function ConvTranspK4(ch_in::Int, ch_out::Int, activation::Function=identity)
    gn = kf * √(w4 * ch_in)
    
    return ConvTranspose((4,4), ch_in => ch_out, activation;
                         stride=2,
                         pad=SamePad(),
                         bias=true,
                         init=kaiming_normal(gain=gn)
    )
end


function DilatedConvK3(ch_in::Int, ch_out::Int, activation::Function=identity;
                       stride::Int=1,
                       dilation::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")
    gn = kf * √(w3 * ch_in)

    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                bias=true,
                dilation=dilation,                # dilation == 1 => ConvK3
                init=kaiming_normal(gain=gn)
)
end
