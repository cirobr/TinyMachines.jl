@info "mobileunet tests..."


# MobileUNet
modelcpu = Chain(MobileUNet(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256,256,1,1) || @error "logits error"


# return features
modelcpu = mobileunet(3, 3; activation=Flux.leakyrelu)
yhat  = modelcpu(x3; return_features=true)
@test size(yhat.encoder.x1)  == (128,128,16,1) || @error "encoder.x1 error"
@test size(yhat.encoder.x2)  == (64,64,24,1)   || @error "encoder.x2 error"
@test size(yhat.encoder.x3)  == (32,32,32,1)   || @error "encoder.x3 error"
@test size(yhat.encoder.x4)  == (16,16,96,1)   || @error "encoder.x4 error"
@test size(yhat.encoder.x5)  == (8,8,1280,1)   || @error "encoder.x5 error"
@test size(yhat.logits) == (256,256,3,1)       || @error "logits error"
