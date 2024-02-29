using ESPNet
using Flux
using Flux: @withprogress, DataLoader, gpu
using CUDA

model = ESPnet(3, 1; K=1) |> gpu

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

# @show ps=sum([length(v) for v in vec.(Flux.params(model))])
X = rand(Float32, (256,256,3,1)) |> gpu
s = size(model(gpu(X)))
Y = rand(Bool, s) |> gpu
data = DataLoader((X,Y); batchsize=1)


# gpu 4s vs cpu 13s
@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
