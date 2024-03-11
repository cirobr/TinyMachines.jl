using TinyMachines
using Flux
using BenchmarkTools

X = rand(Float32, (256,256,3,1))
model = TinyMachines.irb2(3,3,1)   # ch_in, ch_out, expand_ratio
@btime model(X) samples=5 seconds=15 gcsample=true
