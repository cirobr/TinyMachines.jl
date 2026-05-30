function ConvK1(ch_in::Int, ch_out::Int, activation::Function=identity)
    return Conv((1,1), ch_in => ch_out, activation;
                bias=true,
    )
end
PointwiseConv = ConvK1


function ConvK2(ch_in::Int, ch_out::Int, activation::Function=identity)
    return Conv((2,2), ch_in => ch_out, activation;
                stride=2,
                pad=SamePad(),
                bias=true,
    )
end


function ConvK3(ch_in::Int, ch_out::Int, activation::Function=identity;
                stride::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")

    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                bias=true,
    )
end


function ConvTranspK2(ch_in::Int, ch_out::Int, activation::Function=identity;
                      stride::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")
    
    return ConvTranspose((2,2), ch_in => ch_out, activation;
                         stride=stride,
                         pad=SamePad(),
                         bias=true,
    )
end


function ConvTranspK4(ch_in::Int, ch_out::Int, activation::Function=identity)
    return ConvTranspose((4,4), ch_in => ch_out, activation;
                         stride=2,
                         pad=SamePad(),
                         bias=true,
    )
end


function DilatedConvK3(ch_in::Int, ch_out::Int, activation::Function=identity;
                       stride::Int=1,
                       dilation::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")

    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                bias=true,
                dilation=dilation,   # dilation == 1 => ConvK3
)
end
