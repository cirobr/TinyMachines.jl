using Flux
using Flux: DataLoader

struct prelu
    a::Vector
end
Flux.@functor prelu

function (m::prelu)(x)
    return m.a[1] .* (x -> relu(x))
end

model=prelu([1])
Flux.trainable(model)
Flux.params(model)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)
