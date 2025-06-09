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
        )
        for k in 1:K]
    dilated = Chain(chain...)

    return pointwise, dilated
end


# ESPBlock1 is a single ESP block with 1 parallel dilated convolution
# It is a special case of ESPBlock4 with K=1
function ESPBlock1(ch_in::Int, ch_out::Int;   # input/output channels
                   stride::Int=1,             # stride for modulated downsampling
)
    @assert stride ∈ 1:2 || error("stride must be 1 or 2")
    return Chain(
        ConvK1(ch_in, ch_out),                  # pointwise convolution
        ConvK3(ch_out, ch_out; stride=stride),  # d=1 dilated convolution
        BatchNorm(ch_out),
        ConvPReLU(ch_out)
    )
end



# ESPBlock4 is a single ESP block with 4 parallel dilated convolutions
struct ESPBlock4
    pointwise::Conv
    dilated::Chain
end
@layer ESPBlock4

function ESPBlock4(ch_in::Int, ch_out::Int)
    pointwise, dilated = esp(ch_in, ch_out; K=4)
    return ESPBlock4(pointwise, dilated)
end

function (m::ESPBlock4)(x)
    pw = m.pointwise(x)                          # pointwise convolution
    
    sum1 = m.dilated[1](pw)                      # dilated convolutions
    sum2 = m.dilated[2](pw)
    sum3 = m.dilated[3](pw)
    sum4 = m.dilated[4](pw)

    sum2 += sum1                                 # hierarchical sums
    sum3 += sum2
    sum4 += sum3

    yhat = cat(sum1, sum2, sum3, sum4; dims=3)   # concatenate
    return x + yhat                              # residual connection
end

# ESPAlpha is a chain of ESPBlock4 blocks, where alpha is the number of blocks
function ESPAlpha(ch::Int; alpha::Int=1)
    chain = [ESPBlock4(ch, ch) for _ in 1:alpha]
    return Chain(chain...)
end
