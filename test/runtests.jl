using TinyMachines; tm=TinyMachines
using Test
using Flux; import Flux: relu
using Random

Random.seed!(1234)
x3 = randn(Float32, (256,256,3,1))

@testset "TinyMachines.jl" begin
    include("./convlayers-tests.jl")
    include("./unets-tests.jl")
    include("./mobileunet-tests.jl")
    include("./espblocks-tests.jl")
    include("./espnet-tests.jl")
end
