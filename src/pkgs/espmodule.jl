struct ESPmodule
    pointwise
    dilated
    K::Int
    add::Bool
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPmodule(ch_in::Int, ch_out::Int; K::Int=5, add=false)
    # if add && ch_in != ch_out
    #     error("ch_in must equal ch_out when add=true")
    # end

    d = ch_out / K
    # if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = ConvK1(ch_in, d, identity)
    dilated   = [DilatedConvK3(d, d, identity; dilation=2^(k-1)) for k in 1:K]
    dilated   = Chain(dilated...)

    return ESPmodule(pointwise, dilated, K, add)
end


function (m::ESPmodule)(x)
    # pointwise convolution
    pw = m.pointwise(x)

    # dilated convolutions
    sums = map(i -> m.dilated[i](pw), 1:m.K)
    # for i in 2:m.K   sums[i] += sums[i-1]   end

    # concatenate sums
    yhat = cat(sums..., dims=3)

    # add concatenation with input tensor
    if m.add  yhat = x + yhat   end
    return yhat
end

Flux.@functor ESPmodule
