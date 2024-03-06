using Flux
using Flux: DataLoader

struct prelu
    alpha::Vector{Float32}
end

# function prelu(alpha::Vector{Float32})
#     return prelu(alpha)
# end

function (m::prelu)(x)
    return m.alpha[1] .* x
end
Flux.@functor prelu

model=prelu([1])
Flux.trainable(model)
Flux.params(model)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

X = rand(Float32, (256,256,1,1))
Y = rand(Bool, (256,256,1,1))
data = DataLoader((X,Y); batchsize=1)

@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
