using TinyMachines
using Flux
using BenchmarkTools

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = Flux.DataLoader((X,Y); batchsize=1)

model = UNet4(3,1)
@btime model(X) samples=5 seconds=15 gcsample=true


# loss(yhat, y) = Flux.mse(yhat, y)
# opt = Flux.Adam()
# optstate = Flux.setup(opt, model)

# @btime Flux.train!(model, data, optstate) do m,x,y
#     loss(m(x), y)
# end samples=5 seconds=5 gcsample=true
