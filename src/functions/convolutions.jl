### convolutions
const w1 = 1
const w2 = 2 * 2
const w3 = 3 * 3
const w4 = 4 * 4
const kf = 1.f-2


function ConvK1(ch_in::Int, ch_out::Int, activation=identity)
    kgain = kf * √(w1 * ch_in)

    return Conv((1,1), ch_in => ch_out, activation;
                stride=1,
                pad=0,
                bias=false,   # original bias=true
                dilation=1,
                init=kaiming_normal(gain=kgain)
    )
end
PointwiseConv(ch_in::Int, ch_out::Int, activation=identity) = ConvK1(ch_in, ch_out, activation)


function ConvK2(ch_in::Int, ch_out::Int, activation=identity)
    kgain = kf * √(w2 * ch_in)

    return Conv((2,2), ch_in => ch_out, activation;
                stride=2,
                pad=SamePad(),
                bias=true,
                dilation=1,
                init=kaiming_normal(gain=kgain)
    )
end


function ConvK3(ch_in::Int, ch_out::Int, activation=identity;
                stride::Int=1)
    if stride ∉ [1,2]   return error("Stride must be 1 or 2.")   end
    kgain = kf * √(w3 * ch_in)
    
    return Conv((3,3), ch_in => ch_out, activation;
                stride=stride,
                pad=SamePad(),
                bias=true,
                dilation=1,
                init=kaiming_normal(gain=kgain)
    )
end


function DilatedConvK3(ch_in::Int, ch_out::Int, activation=identity;
                       dilation::Int)
    kgain = kf * √(w3 * ch_in)

    return Conv((3,3), ch_in => ch_out, activation;
                stride=1,
                pad=SamePad(),
                bias=true,
                dilation=dilation,
                init=kaiming_normal(gain=kgain)
    )
end


function UpConvK2(ch_in::Int, ch_out::Int, activation=identity)
    return Chain(Upsample(scale=(4,4)), ConvK2(ch_in, ch_out, activation))
end


function ConvTranspK2(ch_in::Int, ch_out::Int, activation=identity;
                      stride::Int=1)
    if stride ∉ [1,2]   return error("Stride must be 1 or 2.")   end
    kgain = kf * √(w2 * ch_in)
    
    return ConvTranspose((2,2), ch_in => ch_out, activation;
                         stride=stride,
                         pad=SamePad(),
                         bias=true,
                         dilation=1,
                         init=kaiming_normal(gain=kgain)
    )
end


function ConvTranspK4(ch_in::Int, ch_out::Int, activation=identity)
    kgain = kf * √(w4 * ch_in)
    
    return ConvTranspose((4,4), ch_in => ch_out, activation;
                         stride=2,
                         pad=SamePad(),
                         bias=true,
                         dilation=1,
                         init=kaiming_normal(gain=kgain)
    )
end


MaxPoolK2 = MaxPool((2,2); pad=0, stride=2)
