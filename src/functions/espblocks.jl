# input image downsampling
downsampling = MeanPool((3,3); pad=SamePad(), stride=2)


# generic ESP block with K dilated convolutions
function esp(ch_in::Int, ch_out::Int;   # input/output channels
             activation,                # activation function
             K::Int                     # number of dilated convolutions
)
    @assert ch_out % K == 0 || error("ch_out must be divisible by K")

    d = ch_out ÷ K
    dils = [2^(k-1) for k in 1:K]  # dilated indices

    act = activation == "prelu" ? PReLU(d) : activation
    pointwise = ConvK1(ch_in, d)
    vector = [Chain(DilatedConvK3(d, d; dilation=dils[k]),
              BatchNorm(d),
              act
              ) for k in 1:K]
    dilated = Chain(vector...)

    return pointwise, dilated
end



# ESPBlock1 is a ESP block with 1 dilated convolution, plus stride for downsampling
struct ESPBlock1
    chain::Conv
end
@layer ESPBlock1

function ESPBlock1(ch_in::Int, ch_out::Int;   # input/output channels
                   activation,                # activation function
                   stride::Int,               # stride for downsampling modulation
)
    @assert stride ∈ 1:2 || error("stride must be 1 or 2")

    act = activation == "prelu" ? PReLU(ch_out) : activation
    return Chain(
        ConvK1(ch_in, ch_out),                  # pointwise convolution
        ConvK3(ch_out, ch_out; stride=stride),  # d=1 dilation & downsampling
        BatchNorm(ch_out),
        act
    )
end

function (m::ESPBlock1)(x)
    yhat = m.chain(x)            # pointwise convolution
    s = size(x) != size(yhat)    # check if downsampling is applied
    return s ? yhat : x + yhat   # no residual connection if downsampling
end



# ESPBlock4 is a ESP block with 4 parallel dilated convolutions, and no stride
struct ESPBlock4
    pointwise::Conv
    dilated::Chain
end
@layer ESPBlock4

function ESPBlock4(ch_in::Int, ch_out::Int; activation)
    pointwise, dilated = esp(ch_in, ch_out, activation=activation, K=4)
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
