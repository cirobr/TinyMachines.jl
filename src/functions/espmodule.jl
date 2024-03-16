struct ESPmodule
    pointwise
    dilated
    K::Int
    add::Bool
end
@layer ESPmodule trainable=(pointwise, dilated)


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPmodule(ch_in::Int, ch_out::Int; K::Int=4, add=false)
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

    return ESPmodule(pointwise, dilated, K, add)
end


function (m::ESPmodule)(x)
    pw   = m.pointwise(x)                                # pointwise convolution
    sums = map(i -> m.dilated[i](pw), 1:m.K)             # dilated convolutions
    map!(i -> sums[i] + sums[i-1], sums[2:end], 2:m.K)   # hierarchical sums
    yhat = cat(sums...; dims=3)                          # concatenate

    if m.add  yhat = x + yhat   end                      # residual connection
    return yhat
end
