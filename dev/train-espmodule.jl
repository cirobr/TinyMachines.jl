using ESPNet
using Flux
using Flux: @withprogress, DataLoader, gpu
using CUDA

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,4,1))
data = DataLoader((X,Y); batchsize=1) |> gpu

model = ESPmodule(3, 4; K=4) |> gpu

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

@show ps=sum([length(v) for v in vec.(Flux.params(model))])

# gpu 0.17s gpu
@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end

