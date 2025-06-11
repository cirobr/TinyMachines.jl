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
        # leakyrelu
    )
end



# ESPBlock4 is a single ESP block with 4 parallel dilated convolutions
function esp4(ch_in::Int, ch_out::Int)   # input/output channels
    K = 4
    dils = [1, 2, 4, 8]
    @assert mod(ch_out, K) == 0 || error("ch_out must be divisible by 4")

    d = ch_out ÷ K
    pointwise = ConvK1(ch_in, d)
    chain1 = Chain( DilatedConvK3(d, d; dilation=dils[1]), BatchNorm(d), ConvPReLU(d) )
    chain2 = Chain( DilatedConvK3(d, d; dilation=dils[2]), BatchNorm(d), ConvPReLU(d) )
    chain3 = Chain( DilatedConvK3(d, d; dilation=dils[3]), BatchNorm(d), ConvPReLU(d) )
    chain4 = Chain( DilatedConvK3(d, d; dilation=dils[4]), BatchNorm(d), ConvPReLU(d) )
    dilated = Chain(chain1, chain2, chain3, chain4)

    return pointwise, dilated
end

struct ESPBlock4
    pointwise::Conv
    dilated::Chain
end
@layer ESPBlock4

function ESPBlock4(ch_in::Int, ch_out::Int)
    pointwise, dilated = esp4(ch_in, ch_out)
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

function ESPAlpha(ch::Int; alpha::Int=1)
    @assert alpha ∈ 1:10 || error("alpha must be between 1 and 10")
    chain = Chain(ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
                  ESPBlock4(ch, ch),
    )
    @assert length(chain) == 10 || error("chain must have 10 blocks")
    return chain[1:alpha]
end





### functions with slow compilation due to recursive chaining ###
"""
# ESP block constructor
K = number of parallel dilated convolutions = height of pyramid
d = number of channels for each parallel dilated convolution
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
"""
