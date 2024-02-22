using TinyMachines
using Flux


struct ESPNet
    chain::Chain
    K::Int
end


# calculates kernel size for the kth dilated convolution with input channel M
n(M::Int, k::Int) = (M-1)*2^(k-1) + 1


# calculates kernel sizes for K parallel dilated convolutions with input channel M
function kernel_sizes(M::Int, K::Int)
    sizes = Vector{Int}(undef, K)
    for i in 1:K
        sizes[i] = n(M, i)
    end
    return sizes
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPNet(ch_in::Int, K::Int)
    d = ch_in / K
    if !isinteger(d)   return error("Number of input channels must be divisible by K.")   end
    d = d |> Int

    ks = kernel_sizes(ch_in, K)
    dilated_convs = [TinyMachines.DilatedConv((ks[i], ks[i]), d, d) for i in 1:K]

    res =  Chain(TinyMachines.PointwiseConv(ch_in, d),
                 Parallel(dilated_convs...)
    )

    return ESPNet(res, K)
end

Flux.@functor ESPNet


ESPNet(8, 4)


function (m::ESPNet)(x)

    return m.chain(x)
end
