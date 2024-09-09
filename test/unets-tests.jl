# UNet2
modelcpu = UNet2()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet2(3, 3; activation=Flux.leakyrelu, alpha=2, verbose=true)
yhat  = modelcpu(x3)
s = size.(yhat[2])
@test s[1] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[1]")
@test s[2] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $s[2]")
@test s[3] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[3]")
# @test s[4] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[4]")
@test s[4] == (256,256,3,1)   || error("Expected (256,256,3,1) but got $s[4]")


# UNet4
modelcpu = UNet4()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet4(3, 3; activation=Flux.leakyrelu, alpha=2, verbose=true)
yhat  = modelcpu(x3)
s = size.(yhat[2])
@test s[1] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[1]")
@test s[2] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $s[2]")
@test s[3] == (64,64,128,1)   || error("Expected (64,64,128,1) but got $s[3]")
@test s[4] == (32,32,256,1)   || error("Expected (32,32,256,1) but got $s[4]")
@test s[5] == (64,64,128,1)   || error("Expected (64,64,128,1) but got $s[5]")
@test s[6] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $s[6]")
@test s[7] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[7]")
# @test s[8] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[8]")
@test s[8] == (256,256,3,1)   || error("Expected (256,256,3,1) but got $s[8]")


# UNet5
modelcpu = UNet5()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet5(3, 3; activation=Flux.leakyrelu, alpha=2, verbose=true)
yhat  = modelcpu(x3)
s = size.(yhat[2])
@test s[1] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[1]")
@test s[2] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $s[2]")
@test s[3] == (64,64,128,1)   || error("Expected (64,64,128,1) but got $s[3]")
@test s[4] == (32,32,256,1)   || error("Expected (32,32,256,1) but got $s[4]")
@test s[5] == (16,16,512,1)   || error("Expected (16,16,512,1) but got $s[5]")
@test s[6] == (32,32,256,1)   || error("Expected (32,32,256,1) but got $s[6]")
@test s[7] == (64,64,128,1)   || error("Expected (64,64,128,1) but got $s[7]")
@test s[8] == (128,128,64,1)  || error("Expected (128,128,64,1) but got $s[8]")
@test s[9] == (256,256,32,1)  || error("Expected (256,256,32,1) but got $s[9]")
# @test s[10] == (256,256,32,1) || error("Expected (256,256,32,1) but got $s[10]")
@test s[10] == (256,256,3,1)  || error("Expected (256,256,3,1) but got $s[10]")
