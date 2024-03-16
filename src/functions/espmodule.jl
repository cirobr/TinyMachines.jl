struct ESPmoduleK1
    pointwise
    dilated
    add::Bool
end
@layer ESPmoduleK1 trainable=(pointwise, dilated)


struct ESPmoduleK4
    pointwise
    dilated
    add::Bool
end
@layer ESPmoduleK4 trainable=(pointwise, dilated)


function ESPmoduleK1(ch_in::Int, ch_out::Int; K::Int=1, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out / K |> Int
    pointwise = ConvK1(ch_in, d, identity)
    dilated   = ConvK3(d, d, identity; stride=1)

    return ESPmoduleK1(pointwise, dilated, add)
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPmoduleK4(ch_in::Int, ch_out::Int; K::Int=4, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = ConvK1(ch_in, d, identity)

    ds = [2^(k-1) for k in 1:K]
    dilated = [DilatedConvK3(d, d, identity; dilation=ds[k]) for k in 1:K]
    dilated = Chain(dilated...)

    return ESPmoduleK4(pointwise, dilated, add)
end


function (m::ESPmoduleK1)(x)
    pw   = m.pointwise(x)                        # pointwise convolution
    yhat = m.dilated(pw)                         # dilated convolutions
    if m.add  yhat = x + yhat   end              # residual connection
    return yhat
end


function (m::ESPmoduleK4)(x)
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
