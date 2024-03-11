struct ESPmoduleK1
    pointwise
    dilated
    K::Int
    add::Bool
end
@layer ESPmoduleK1 trainable=(pointwise, dilated)


struct ESPmoduleK4
    pointwise
    dilated
    K::Int
    add::Bool
end
@layer ESPmoduleK4 trainable=(pointwise, dilated)


function ESPmoduleK1(ch_in::Int, ch_out::Int; K::Int=1, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out
    pointwise = ConvK1(ch_in, d, identity)
    dilated   = DilatedConvK3(d, d, identity; dilation=1)

    return ESPmoduleK1(pointwise, dilated, K, add)
end


function ESPmoduleK4(ch_in::Int, ch_out::Int; K::Int=4, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = ConvK1(ch_in, d, identity)
    dilated   = Chain(DilatedConvK3(d, d, identity; dilation=2^(1-1)),
                      DilatedConvK3(d, d, identity; dilation=2^(2-1)),
                      DilatedConvK3(d, d, identity; dilation=2^(3-1)),
                      DilatedConvK3(d, d, identity; dilation=2^(4-1)))

    return ESPmoduleK4(pointwise, dilated, K, add)
end


function (m::ESPmoduleK1)(x)
    pw   = m.pointwise(x)                            # pointwise convolution
    yhat = m.dilated(pw)                             # dilated convolutions
    if m.add  yhat = x + yhat   end                  # residual connection
    return yhat
end


function (m::ESPmoduleK4)(x)
    pw    = m.pointwise(x)                           # pointwise convolution
    
    sums1 = m.dilated[1](pw)                         # dilated convolutions
    sums2 = m.dilated[2](pw)
    sums3 = m.dilated[3](pw)
    sums4 = m.dilated[4](pw)

    sums2 = sums2 + sums1                            # hierarchical sum
    sums3 = sums3 + sums2
    sums4 = sums4 + sums3

    yhat = cat(sums1, sums2, sums3, sums4, dims=3)   # concatenation

    if m.add  yhat = x + yhat   end                  # residual connection
    return yhat
end
