using TinyMachines
using Flux
using Flux: DataLoader

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = DataLoader((X,Y); batchsize=1)

model = ESPNet(3,1; activation=prelu)

loss(yhat, y) = Flux.mse(yhat, y)
opt = Flux.Adam()
optstate = Flux.setup(opt, model)

@time Flux.train!(model, data, optstate) do m,x,y
    loss(m(x), y)
end
