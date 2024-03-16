using TinyMachines; tm=TinyMachines
using Flux
using Random
using BenchmarkTools


Random.seed!(1234)
X = rand(Float32, (256,256,64,1))

@info "ConvK1"
model = tm.ConvK1(64,64)
@btime model(X) samples=5 seconds=15 gcsample=true;

@info "ConvK3"
model = tm.ConvK3(64,64)
@btime model(X) samples=5 seconds=15 gcsample=true;

@info "Conv"
model = Flux.Conv((3,3), 64 => 64)
@btime model(X) samples=5 seconds=15 gcsample=true;


@info "done!"