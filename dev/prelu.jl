using Flux
using Flux: DataLoader

fn_prelu(x, alpha) = x>0 ? x : alpha * x

struct model_prelu
    alpha::Vector{Float32}
end

function (m::model_prelu)(x)
    return fn_prelu.(x, m.alpha[1])
end
Flux.@functor model_prelu

prelu=model_prelu([0.f0])



display(prelu.alpha)
# Flux.trainable(prelu)
Flux.params(prelu)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, prelu)

X = randn(Float32, (256,256,1,1))
Y = rand(Bool, (256,256,1,1))
data = DataLoader((X,Y); batchsize=1)

for i in 1:100 Flux.train!(prelu, data, optstate) do m,x,y
    loss(m(x), y)
end end

prelu.alpha
