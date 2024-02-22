using TinyMachines; tm=TinyMachines
using Flux


struct ESPmodule
    chain::Chain
    K::Int
end


"""
K = number of parallel dilated convolutions = height of pyramid
d = number of input/output channels for all parallel dilated convolutions
"""
function ESPmodule(ch_in::Int, K::Int)
    d = 2^(K-1)
    dilated_convs = [tm.DilatedConvK3(d, d, identity; dilation=2^(i-1)) for i in 1:K]

    res =  Chain(tm.PointwiseConv(ch_in, d),
                 Parallel(dilated_convs...)
    )

    # display(res)
    return ESPmodule(res, K)
end

Flux.@functor ESPmodule


function (m::ESPmodule)(x)

    return m.chain[1](x)
end



x=rand(Float32, (64,64,8,1))
model = ESPmodule(8,5)
model(x)
