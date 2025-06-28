# input image downsampling
downsampling = MeanPool((3,3); pad=SamePad(), stride=2)


# ESPBlock1 is a single ESP block with 1 dilated convolution, plus stride for downsampling
function ESPBlock1(ch_in::Int, ch_out::Int;   # input/output channels
                   stride::Int=1,             # stride for modulated downsampling
)
    @assert stride ∈ 1:2 || error("stride must be 1 or 2")
    return Chain(
        ConvK1(ch_in, ch_out),                  # pointwise convolution
        ConvK3(ch_out, ch_out; stride=stride),  # d=1 dilation & downsampling
        BatchNorm(ch_out),
        ConvPReLU(ch_out)
        # leakyrelu
    )
end


# ESPBlock4 is a single ESP block with 4 parallel dilated convolutions, and no stride
function esp4(ch_in::Int, ch_out::Int)   # input/output channels
    K = 4
    dils = [1, 2, 4, 8]
    @assert ch_out % K == 0 || error("ch_out must be divisible by 4")

    d = ch_out ÷ K
    pointwise = ConvK1(ch_in, d)
    vector = [Chain(DilatedConvK3(d, d; dilation=dils[k]),
              BatchNorm(d),
              ConvPReLU(d)
              # leakyrelu
              ) for k in 1:K]
    dilated = Chain(vector...)

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
    vector = [ESPBlock4(ch, ch) for _ in 1:alpha]
    return Chain(vector...)
end
