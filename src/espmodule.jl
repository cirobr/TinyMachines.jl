using TinyMachines; tm=TinyMachines
using Flux


struct ESPmod
    pointwise
    dilated
    K::Int
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPmod(ch_in::Int, ch_out::Int, K::Int)
    d = ch_out / K
    if !isinteger(d)   error("ch_out must be divisible by K")   end
    d = d |> Int

    pointwise = tm.PointwiseConv(ch_in, d)
    dilated_convs = [tm.DilatedConvK3(d, d, identity; dilation=2^(k-1)) for k in 1:K]
    dilated   = Chain(dilated_convs...)

    return ESPmod(pointwise, dilated, K)
end


function (m::ESPmod)(x)
    # pointwise convolution
    pw = m.pointwise(x)

    # dilated convolutions
    h, w, C = size(pw)[1:4]
    dconvs = Array{Float32}(undef, (h, w, C, m.K))
    for i in 1:m.K   dconvs[:,:,:,i] = m.dilated[i](pw)   end

    # sums of dilated convolutions
    sums = Array{Float32}(undef, (h, w, C, m.K))
    sums[:,:,:,1] = dconvs[:,:,:,1]   
    for i in 2:m.K   sums[:,:,:,i] = dconvs[:,:,:,i] + sums[:,:,:,i-1]   end

    # concatenate sums
    cat_sums = reshape(sums, (h, w, C*m.K, 1))
    size(cat_sums)

    # sum concatenation with input tensor

    
end

Flux.@functor ESPmod


x = rand(Float32, (256,256,3,1))
c1  = tm.ConvK3(3,16; stride=2)(x)
ds1 = tm.ConvK3(3,3;  stride=2)(x)
ct1 = cat(c1, ds1, dims=3)        # 19ch OK
yhat = ESPmod(19, 64, 4)(ct1)     # 64ch OK
