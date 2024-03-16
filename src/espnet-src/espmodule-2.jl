struct ESPmoduleK1
    pointwise
    dilated
    add::Bool
end
@layer ESPmoduleK1 trainable=(pointwise, dilated)


struct ESPmodule
    pointwise
    dilated
    K::Int
    add::Bool
end
@layer ESPmodule trainable=(pointwise, dilated)


function ESPmoduleK1(ch_in::Int, ch_out::Int; K::Int=1, add=false)
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out / K |> Int
    pointwise = ConvK1(ch_in, d, identity)
    dilated   = ConvK3(d, d, identity; stride=1)

    return ESPmoduleK1(pointwise, dilated, add)
end


function (m::ESPmoduleK1)(x)
    pw   = m.pointwise(x)                            # pointwise convolution
    yhat = m.dilated(pw)                             # dilated convolutions
    if m.add  yhat = x + yhat   end                  # residual connection
    return yhat
end


function ESPmodule(ch_in::Int, ch_out::Int; K::Int=4, add=false)
    if K == 1   error("K must be greater than 1, use ESPmoduleK1 instead")   end

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
    pw    = m.pointwise(x)                        # pointwise convolution
    sums = Vector{Array{Float32,4}}(undef, m.K)   # dilated convolutions
    sums = map(i -> m.dilated[i](pw), 1:m.K)
    # hierarchical sum
    yhat = cat(sums...; dims=3)                   # concatenate

    if m.add  yhat = x + yhat   end               # residual connection
    return yhat
end
