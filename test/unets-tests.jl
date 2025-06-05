# UNet4
modelcpu = UNet4()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = unet4(3, 3; activation=Flux.leakyrelu, alpha=2)
yhat  = modelcpu(x3)
@test size(yhat[1]) == (256,256,32,1)  || error("Expected (256,256,32,1) but got $yhat[1]")
@test size(yhat[2]) == (128,128,64,1)  || error("Expected (128,128,64,1) but got $yhat[2]")
@test size(yhat[3]) == (64,64,128,1)   || error("Expected (64,64,128,1) but got $yhat[3]")
@test size(yhat[4]) == (32,32,256,1)   || error("Expected (32,32,256,1) but got $yhat[4]")
@test size(yhat[5]) == (64,64,128,1)   || error("Expected (64,64,128,1) but got $yhat[5]")
@test size(yhat[6]) == (128,128,64,1)  || error("Expected (128,128,64,1) but got $yhat[6]")
@test size(yhat[7]) == (256,256,32,1)  || error("Expected (256,256,32,1) but got $yhat[7]")
@test size(yhat[8]) == (256,256,3,1)   || error("Expected (256,256,3,1) but got $yhat[8]")


# UNet5
modelcpu = UNet5()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = unet5(3, 3; activation=Flux.leakyrelu, alpha=2)
yhat  = modelcpu(x3)
@test size(yhat[1]) == (256,256,32,1)  || error("Expected (256,256,32,1) but got $yhat[1]")
@test size(yhat[2]) == (128,128,64,1)  || error("Expected (128,128,64,1) but got $yhat[2]")
@test size(yhat[3]) == (64,64,128,1)   || error("Expected (64,64,128,1) but got $yhat[3]")
@test size(yhat[4]) == (32,32,256,1)   || error("Expected (32,32,256,1) but got $yhat[4]")
# @test yhat[5] == (16,16,512,1)   || error("Expected (16,16,512,1) but got $yhat[5]")
# @test yhat[6] == (32,32,256,1)   || error("Expected (32,32,256,1) but got $yhat[6]")
# @test yhat[7] == (64,64,128,1)   || error("Expected (64,64,128,1) but got $yhat[7]")
# @test yhat[8] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $yhat[8]")
# @test yhat[9] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $yhat[9]")
# @test yhat[10] == (256,256,3,1)  || error("Expected (256,256,3,1) but got $yhat[10]")
