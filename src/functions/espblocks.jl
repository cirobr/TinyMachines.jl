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
        # ConvPReLU(ch_out)
        leakyrelu
    )
end



# ESPBlock4 is a single ESP block with 4 parallel dilated convolutions
function esp4(ch_in::Int, ch_out::Int)   # input/output channels
    K = 4
    dils = [1, 2, 4, 8]
    @assert mod(ch_out, K) == 0 || error("ch_out must be divisible by 4")

    d = ch_out ÷ K
    pointwise = ConvK1(ch_in, d)
    chain1 = Chain( DilatedConvK3(d, d; dilation=dils[1]), BatchNorm(d), leakyrelu )
    chain2 = Chain( DilatedConvK3(d, d; dilation=dils[2]), BatchNorm(d), leakyrelu )
    chain3 = Chain( DilatedConvK3(d, d; dilation=dils[3]), BatchNorm(d), leakyrelu )
    chain4 = Chain( DilatedConvK3(d, d; dilation=dils[4]), BatchNorm(d), leakyrelu )
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

function ChainedESPBlock4(ch::Int; alpha::Int=1)
    @assert alpha ∈ 1:10 || error("alpha must be in the range 1:10")
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
    # @assert length(chain) == 10 || error("ChainedESPBlock4 must have 10 blocks")
    return chain[1:alpha]
end
