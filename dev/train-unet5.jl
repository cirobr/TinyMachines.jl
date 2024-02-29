using TinyMachines; const tm = TinyMachines
using ESPNet
using Flux
using Flux: @withprogress, DataLoader, gpu
using CUDA

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = DataLoader((X,Y); batchsize=1) |> gpu

model = tm.UNet5(3,1) |> gpu

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

@show ps=sum([length(v) for v in vec.(Flux.params(model))])

# gpu 0.5s vs cpu 5s
@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
