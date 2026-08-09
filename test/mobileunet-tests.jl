@info "mobileunet tests..."


# MobileUNet
modelcpu = Chain(MobileUNet(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256,256,1,1) || @error "logits error"

# return features
modelcpu = mobileunet(3, 3; activation=Flux.leakyrelu)
fs = Flux.activations(modelcpu.d, x3)
@test size(fs[1]) == (128,128,16,1) || @error "encoder x1 error"
@test size(fs[2]) == (64,64,24,1)   || @error "encoder x2 error"
@test size(fs[3]) == (32,32,32,1)   || @error "encoder x3 error"
@test size(fs[4]) == (16,16,96,1)   || @error "encoder x4 error"
@test size(fs[5]) == (8,8,1280,1)   || @error "encoder x5 error"
yhat = modelcpu(x3)
@test size(yhat) == (256,256,3,1)   || @error "logits error"
