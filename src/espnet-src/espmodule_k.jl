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


function DilatedConvK3(ch_in::Int, ch_out::Int, activation=identity;
                       dilation::Int)
    kgain = kf * √(w3 * ch_in)

    return Conv((3,3), ch_in => ch_out, activation;
                stride=1,
                pad=SamePad(),
                bias=true,
                dilation=dilation,
                groups=4,
                init=kaiming_normal(gain=kgain)
    )
end


function ESPmoduleK1(ch_in::Int, ch_out::Int; K::Int=1, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out
    pointwise = ConvK1(ch_in, d, identity)
    dilated   = ConvK3(d, d, identity)

    return ESPmoduleK1(pointwise, dilated, K, add)
end


function ESPmoduleK4(ch_in::Int, ch_out::Int; K::Int=4, add=false)
    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    pointwise = ConvK1(ch_in, d, identity)
    dilations = [2^(k-1) for k in 1:K]
    dilated   = Chain(DilatedConvK3(d, d, identity; dilation=dilations[1]),
                      DilatedConvK3(d, d, identity; dilation=dilations[2]),
                      DilatedConvK3(d, d, identity; dilation=dilations[3]),
                      DilatedConvK3(d, d, identity; dilation=dilations[4]))

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
