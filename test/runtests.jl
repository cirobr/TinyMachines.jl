using Pkg
Pkg.activate(".")
using TinyMachines
using Flux
using CUDA
using Random
using Test

Random.seed!(1234)
x  = randn(Float32, (8,8,1,1))
x1 = randn(Float32, (256,256,1,1))
x3 = randn(Float32, (256,256,3,1))

@testset "TinyMachines.jl" begin
    include("./activations-tests.jl")
    include("./convolutions-tests.jl")   # TODO: DilatedConv
    include("./unets-tests.jl")
    include("./mobileunet-tests.jl")
    include("./espnet-tests.jl")
end
