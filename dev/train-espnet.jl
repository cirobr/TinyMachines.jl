using TinyMachines
using Flux
using BenchmarkTools
using CUDA

X = rand(Float32, (256,256,3,1)) |> gpu
Y = rand(Bool, (256,256,1,1)) |> gpu
data = Flux.DataLoader((X,Y); batchsize=1) |> gpu

model = ESPNet(3,1; activation=relu, K=4) |> gpu
@btime model(X) samples=5 seconds=15 gcsample=true


loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end

# @btime Flux.train!(model, data, optstate) do m,x,y
#     loss(m(x), y)
# end samples=5 seconds=60 gcsample=true
