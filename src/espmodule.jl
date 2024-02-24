struct ESPmodule
    pointwise
    dilated
    K::Int
    add::Bool
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
add = true requires ch_in == ch_out (not checked by code)
"""
function ESPmodule(ch_in::Int, ch_out::Int, K::Int; add=false)
    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = ConvK1(ch_in, d)
    dilated   = [DilatedConvK3(d, d; dilation=2^(k-1)) for k in 1:K]
    dilated   = Chain(dilated...)

    return ESPmodule(pointwise, dilated, K, add)
end


function (m::ESPmodule)(x)
    # pointwise convolution
    pw = m.pointwise(x)

    # dilated convolutions
    h, w, C = size(pw)[1:3]
    dconvs = Array{Float32}(undef, (h, w, C, m.K))
    for i in 1:m.K   dconvs[:,:,:,i] = m.dilated[i](pw)   end

    # sums of dilated convolutions
    sums = Array{Float32}(undef, (h, w, C, m.K))
    sums[:,:,:,1] = dconvs[:,:,:,1]   
    for i in 2:m.K   sums[:,:,:,i] = dconvs[:,:,:,i] + sums[:,:,:,i-1]   end

    # concatenate sums
    cat_sums = reshape(sums, (h, w, C*m.K, 1))

    # sum concatenation with input tensor
    if m.add  return x + cat_sums   end
    return cat_sums
end

Flux.@functor ESPmodule
