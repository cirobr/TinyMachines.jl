using BenchmarkTools
using TinyMachines; tm=TinyMachines
using Flux
using Random
using Test

Random.seed!(1234)
x1 = randn(Float32, (256,256,1,1))
x3 = randn(Float32, (256,256,3,1))

m1 = tm.ConvK3(3,1)
@time m1(x3);

model = ESPmodule(3,4; K=4)
@time (model(x3));