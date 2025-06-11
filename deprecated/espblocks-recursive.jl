### functions with slow compilation due to recursive chaining ###
# ESP block constructor
# K = number of parallel dilated convolutions = height of pyramid
# d = number of channels for each parallel dilated convolution
function esp(ch_in::Int, ch_out::Int;   # input/output channels
             K::Int=1,                  # number of parallel dilated convolutions
)
    @assert K > 1               || error("K must be greater than 1")
    @assert mod(ch_out, K) == 0 || error("ch_out must be divisible by K")

    d = ch_out ÷ K
    dils = [2^(k-1) for k in 1:K]

    pointwise = ConvK1(ch_in, d)
    chain =
        [Chain(DilatedConvK3(d, d; dilation=dils[k]),
               BatchNorm(d),
               ConvPReLU(d)
            #    leakyrelu
        )
        for k in 1:K]
    dilated = Chain(chain...)

    return pointwise, dilated
end

# ESPAlpha is a chain of ESPBlock4 blocks, where alpha is the number of blocks
function ESPAlpha(ch::Int; alpha::Int=1)
    chain = [ESPBlock4(ch, ch) for _ in 1:alpha]
    return Chain(chain...)
end
