using TinyMachines
using Flux
using BenchmarkTools

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = Flux.DataLoader((X,Y); batchsize=1)

model = Conv((1,1), 3=>1, relu)
@btime model(X) samples=5 seconds=15 gcsample=true


# loss(yhat, y) = Flux.mse(yhat, y)
# opt = Flux.Adam()
# optstate = Flux.setup(opt, model)

# @btime Flux.train!(model, data, optstate) do m,x,y
#     loss(m(x), y)
# end samples=5 seconds=5 gcsample=true
