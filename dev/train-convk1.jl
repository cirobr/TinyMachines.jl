using Pkg
Pkg.activate( expanduser("~/envs/dev"))
using Flux
using CUDA

model = Flux.Conv((1,1), 3=>1) |> gpu

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

X = rand(Float32, (128,128,3,1)) |> gpu
s = size(model(X))
Y = rand(Bool, s) |> gpu
data = Flux.DataLoader((X,Y); batchsize=1)

@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
