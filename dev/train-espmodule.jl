using TinyMachines
using Flux
using BenchmarkTools

X = rand(Float32, (256,256,64,1))
model = ESPmoduleK4(64,64)
@btime model(X) samples=5 seconds=15 gcsample=true;
