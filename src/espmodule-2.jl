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
    if add && ch_in != ch_out
        error("ch_in must equal ch_out when add=true")
    end

    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = ConvK1(ch_in, d, identity)
    dilated   = [DilatedConvK3(d, d, identity; dilation=2^(k-1)) for k in 1:K]
    dilated   = Chain(dilated...)

    return ESPmodule(pointwise, dilated, K, add)
end


function (m::ESPmodule)(x)
    # pointwise convolution
    pw = m.pointwise(x)
    h, w, C, N = size(pw)

    # dilated convolutions
    sums = Array{Float32}(undef, h, w, C, N, m.K)
    if occursin("CuArray", string(typeof(x)))    sums = sums |> gpu   end
    # @show eltype(x), typeof(x)
    # @show eltype(sums), typeof(sums)

    for i in 1:m.K
        sums[:, :, :, :, i] = m.dilated[i](pw)
    end
    for i in 2:m.K
        sums[:, :, :, :, i] += sums[:, :, :, :, i-1]
    end

    # concatenation
    yhat = reshape(sums, h, w, C*m.K, N)

    # add concatenation with input tensor
    if m.add  yhat = x + yhat   end

    return yhat
end

Flux.@functor ESPmodule
