using Flux
using Flux: Dense, relu, DataLoader, Conv

X = rand(Float32, (256,256,3,1))
Y = rand(Bool, (256,256,1,1))
data = DataLoader((X,Y); batchsize=1)

prelu3 = Conv((1,1), 1=>1, relu, bias=false, init=ones)
opt = Flux.Adam()
optstate = Flux.setup(opt, prelu3)


loss(yhat, y) = Flux.mse(yhat, y)
@time Flux.train!(prelu3, data, optstate) do m,x,y
    loss(m(x), y)
end
