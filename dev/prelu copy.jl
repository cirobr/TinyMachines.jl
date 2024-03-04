using Flux
using Flux: DataLoader

struct prelu
    a::Vector
end

function (m::prelu)(x)
    return alpha[1] * Flux.relu(x)
end
Flux.@functor prelu

model=prelu([1])
# Flux.trainable(model.a)
Flux.params(model)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)
