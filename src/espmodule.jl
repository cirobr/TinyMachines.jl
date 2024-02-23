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
function ESPmod(ch_in::Int, K::Int)
    d = 2^(K-1)
    dilated_convs = [tm.DilatedConvK3(d, d, identity; dilation=2^(i-1)) for i in 1:K]
    dilated   = Chain(dilated_convs...)
    pointwise = tm.PointwiseConv(ch_in, d)

    return ESPmod(pointwise, dilated, K)
end

Flux.@functor ESPmod


function (m::ESPmod)(x)
    pw = m.pointwise(x)

    h, w, C = size(pw)[1:3]
    convs = Array{Float32}(undef, (h, w, C, m.K))
    for i in 1:m.K   convs[:,:,:,i] = m.dilated[i](pw)   end

    sums = Array{Float32}(undef, (h, w, C, m.K))
    sums[:,:,:,1] = convs[:,:,:,1]
    for i in 2:m.K   sums[:,:,:,i] = convs[:,:,:,i] + sums[:,:,:,i-1]   end

    @show size(x)
    @show size(pw)
    @show size(convs)
    @show size(sums)
end



x=rand(Float32, (64,64,8,1))
model = ESPmod(8,5)
model(x)
