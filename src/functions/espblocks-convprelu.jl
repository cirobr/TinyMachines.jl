struct ESPBlock1
    pointwise
    dilated
    add::Bool
end
@layer ESPBlock1 trainable=(pointwise, dilated)


struct ESPBlock4
    pointwise
    dilated
    add::Bool
end
@layer ESPBlock4 trainable=(pointwise, dilated)


function ESPBlock1(ch_in::Int, ch_out::Int;
                   stride::Int=1,   # downsampling modulated by stride
                   add::Bool=false)
    # K = 1
    # @assert mod(ch_out, K) == 0 || error("ch_out must be divisible by K")

    d = ch_out

    pointwise = ConvK1(ch_in, d)
    dilated   = Chain(ConvK3(d, d; stride=stride),
                      BatchNorm(d),
                      ConvPReLU(d)
    )

    return ESPBlock1(pointwise, dilated, add)
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPBlock4(ch_in::Int, ch_out::Int;
                   # no stride, no downsampling
                   add::Bool=false)
    K = 4
    @assert mod(ch_out, K) == 0 || error("ch_out must be divisible by K")

    d = ch_out ÷ K
    dils = [2^(k-1) for k in 1:K]

    pointwise = ConvK1(ch_in, d)
    dilated_vec =
        [Chain(DilatedConvK3(d, d; stride=1, dilation=dils[k]),
               BatchNorm(d),
               ConvPReLU(d)
        )
        for k in 1:K]
    dilated = Chain(dilated_vec...)

    return ESPBlock4(pointwise, dilated, add)
end


function (m::ESPBlock1)(x)
    pw   = m.pointwise(x)                        # pointwise convolution
    yhat = m.dilated(pw)                         # dilated convolutions
    if m.add  yhat = x + yhat   end              # residual connection
    return yhat
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

    if m.add  yhat = x + yhat   end              # residual connection
    return yhat
end


function ESPBlock4_alpha(ch::Int; alpha::Int=1)
    chain = [ESPBlock4(ch, ch; add=true) for k in 1:alpha]
    return Chain(chain...)
end
