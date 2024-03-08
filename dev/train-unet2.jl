using TinyMachines
using Flux
using BenchmarkTools

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = Flux.DataLoader((X,Y); batchsize=1)

model = UNet2(3,1)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

@btime model(X);

@btime Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
