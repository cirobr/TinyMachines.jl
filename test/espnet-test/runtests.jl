using TinyMachines
using Flux: relu
using Test

@testset "espnet" begin
    x = rand(Float32, (256,256,3,4))
    m = ESPmoduleK4(3, 4)
    yhat = m(x)
    @test size(yhat) == (256,256,4,4) || @error "size(yhat) == $(size(yhat))"

    x = rand(Float32, (256,256,4,4))
    m = ESPmoduleK4(4, 4; add=true)
    yhat = m(x)
    @test size(yhat) == (256,256,4,4) || @error "size(yhat) == $(size(yhat))"

    x = rand(Float32, (256,256,3,1))
    m = ESPmoduleK1(3, 1)
    yhat = m(x)
    @test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"

    x = rand(Float32, (256,256,3,1))
    m = ESPNet(3,1; activation=relu)
    yhat = m(x)
    @test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"
end
