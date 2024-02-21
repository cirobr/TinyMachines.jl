"""
He, K., Zhang, X., Ren, S., Sun, J.: Delving deep into rectifiers: Surpassing human-level
performance on imagenet classification.

https://arxiv.org/abs/1502.01852
"""
##### PReLU version 1
prelu1 = Scale(1, relu, bias=false)


##### PReLU version 2
struct PReLU
    scale::Vector{Float32}
end

PReLU(scale::Float32=1.f0) = PReLU([scale])

function (p::PReLU)(x)
    return relu(p.scale .* x)
end

@functor PReLU

prelu2 = PReLU()
