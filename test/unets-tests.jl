@info "unets tests..."


# UNet4
modelcpu = Chain(UNet4(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1) || @error "logits error"

# return features
modelcpu = unet4(3, 3; activation=Flux.leakyrelu, alpha=2)
fs = Flux.activations(modelcpu.encoder, x3)
@test size(fs[1]) == (256,256,32,1) || @error "encoder 1 error"
@test size(fs[2]) == (128,128,64,1) || @error "encoder 2 error"
@test size(fs[3]) == (64,64,128,1)  || @error "encoder 3 error"
@test size(fs[4]) == (32,32,256,1)  || @error "encoder 4 error"
yhat  = modelcpu(x3)
@test size(yhat)  == (256,256,3,1)  || @error "logits error"


# UNet5
modelcpu = Chain(UNet5(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1) || @error "logits error"

modelcpu = unet5(3, 3; activation=Flux.leakyrelu, alpha=2)
fs = Flux.activations(modelcpu.encoder, x3)
@test size(fs[1]) == (256,256,32,1) || @error "encoder 1 error"
@test size(fs[2]) == (128,128,64,1) || @error "encoder 2 error"
@test size(fs[3]) == (64,64,128,1)  || @error "encoder 3 error"
@test size(fs[4]) == (32,32,256,1)  || @error "encoder 4 error"
@test size(fs[5]) == (16,16,512,1)  || @error "encoder 5 error"
yhat  = modelcpu(x3)
@test size(yhat)  == (256,256,3,1)  || @error "logits error"
