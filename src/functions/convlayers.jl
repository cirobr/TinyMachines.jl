function Conv1(ch_in::Int, ch_out::Int, activation::Function=identity)
    return Conv((1,1), ch_in => ch_out, activation;
                init=kaiming_normal(),
    )
end
PointwiseConv = Conv1


function Conv2(ch_in::Int, ch_out::Int, activation::Function=identity)
    return Conv((2,2), ch_in => ch_out, activation;
                stride=2,
                pad=SamePad(),
                init=kaiming_normal(),
    )
end


function Conv3(ch_in::Int, ch_out::Int, activation::Function=identity;
                stride::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")
    
    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                init=kaiming_normal(),
    )
end


function ConvTr2(ch_in::Int, ch_out::Int, activation::Function=identity)
    return ConvTranspose((2,2), ch_in => ch_out, activation;
                        stride=2,
                        pad=SamePad(),
                        init=kaiming_normal(),
    )
end


function ConvTr4(ch_in::Int, ch_out::Int, activation::Function=identity)
    return ConvTranspose((4,4), ch_in => ch_out, activation;
                        stride=2,
                        pad=SamePad(),
                        init=kaiming_normal(),
    )
end


function DilatedConv3(ch_in::Int, ch_out::Int, activation::Function=identity;
                       stride::Int=1,
                       dilation::Int=1)
    @assert stride ∈ [1,2] || error("Stride must be 1 or 2.")

    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                dilation=dilation,
                init=kaiming_normal(),
)
end
