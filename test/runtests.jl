using TinyMachines; tm=TinyMachines
using Test
using Flux; import Flux: relu
using Random

Random.seed!(1234)
x3 = randn(Float32, (256,256,3,1))
x4 = randn(Float32, (256,256,4,1))
x8 = randn(Float32, (256,256,8,1))

@info "TinyMachines.jl tests..."
@testset "TinyMachines.jl" begin
    include("./activationlayers-tests.jl")
    include("./convlayers-tests.jl")
    include("./espblocks-tests.jl")
    ### TODO: irblocks-tests

    include("./espnet-tests.jl")
    include("./mobileunet-tests.jl")
    include("./unets-tests.jl")

    # include("./epoch-tests.jl")
end
