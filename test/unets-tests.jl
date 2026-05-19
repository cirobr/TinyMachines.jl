@info "unets tests..."


# UNet4
modelcpu = Chain(UNet4(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1) || @error "logits error"


# return features
modelcpu = unet4(3, 3; activation=Flux.leakyrelu, alpha=2)
yhat  = modelcpu(x3; return_features=true)
@test size(yhat.encoder.enc1) == (256,256,32,1) || @error "encoder.enc1 error"
@test size(yhat.encoder.enc2) == (128,128,64,1) || @error "encoder.enc2 error"
@test size(yhat.encoder.enc3) == (64,64,128,1)  || @error "encoder.enc3 error"
@test size(yhat.encoder.enc4) == (32,32,256,1)  || @error "encoder.enc4 error"
@test size(yhat.logits)       == (256,256,3,1)  || @error "logits error"


# UNet5
modelcpu = Chain(UNet5(3,1), sigmoid)
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1) || @error "logits error"


# return features
modelcpu = unet5(3, 3; activation=Flux.leakyrelu, alpha=2)
yhat  = modelcpu(x3; return_features=true)
@test size(yhat.encoder.enc1) == (256,256,32,1) || @error "encoder.enc1 error"
@test size(yhat.encoder.enc2) == (128,128,64,1) || @error "encoder.enc2 error"
@test size(yhat.encoder.enc3) == (64,64,128,1)  || @error "encoder.enc3 error"
@test size(yhat.encoder.enc4) == (32,32,256,1)  || @error "encoder.enc4 error"
@test size(yhat.encoder.enc5) == (16,16,512,1)  || @error "encoder.enc5 error"
@test size(yhat.logits)       == (256,256,3,1)  || @error "logits error"
