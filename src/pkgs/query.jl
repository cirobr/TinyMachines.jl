using Flux: Chain, @functor, Conv, relu
struct Mymodel
    chain
    N
end

function Mymodel(ch_in, ch_out, N)
    models = [Conv((1,1), ch_in => ch_out, relu) for i in 1:N]
    chain = Chain(models...)

    return Mymodel(chain, N)
end

function (m::Mymodel)(x)
    res =  map(i -> m.chain[i](x), 1:m.N)
    return cat(res..., dims=3)
end

@functor Mymodel

x=rand(Float32, 32, 32, 3, 1)
model = Mymodel(3, 64, 5)
model(x)