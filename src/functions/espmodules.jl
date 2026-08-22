# input image downsampling
# downsampling = MeanPool((3,3); pad=SamePad(), stride=2)
downsampling = MeanPool((2,2); pad=SamePad(), stride=2)
img_ds1(x) = downsampling(x)                       # downsampling stage-1 input image
img_ds2(x) = downsampling(x[:, :, end-2:end, :])   # downsampling stage-2 input image (last-3 channels of stage-1 output)


# generic ESP module with K dilated convolutions
function esp(
    ch_in::Int, ch_out::Int;   # input/output channels
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



# ESP1 is a ESP module with one dilated convolution, plus stride for downsampling
struct ESP1
    chain::Chain
end
@layer ESP1

function ESP1(
    ch_in::Int, ch_out::Int;   # input/output channels
    activation,                # activation function
    stride::Int,               # stride for downsampling modulation
)
    @assert stride ∈ 1:2 || error("stride must be 1 or 2")
    act = activation == "prelu" ? PReLU(ch_out) : activation

    chain = Chain(
        ConvK1(ch_in, ch_out),
        ConvK3(ch_out, ch_out; stride=stride),
        BatchNorm(ch_out),
        act
    )
    return ESP1(chain)
end

function (m::ESP1)(x)
    yhat = m.chain(x)
    if size(x) == size(yhat)
        return x + yhat
    else
        return yhat
    end
end


# ESP4 is a ESP module with 4 parallel dilated convolutions, and no stride
struct ESP4
    pointwise::Conv
    dilated::Chain
end
@layer ESP4

function ESP4(ch_in::Int, ch_out::Int; activation)
    pointwise, dilated = esp(ch_in, ch_out, activation=activation, K=4)
    return ESP4(pointwise, dilated)
end

function (m::ESP4)(x)
    pw = m.pointwise(x)                    # pointwise convolution
    
    d1 = m.dilated[1](pw)                  # dilated convolutions
    d2 = m.dilated[2](pw) + d1
    d3 = m.dilated[3](pw) + d2
    d4 = m.dilated[4](pw) + d3

    yhat = cat(d1, d2, d3, d4; dims=3)     # concatenate
    return x + yhat                        # residual connection
end
