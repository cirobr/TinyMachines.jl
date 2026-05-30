@info "espnet tests..."


# logits output
m = Chain(ESPNet(3,1), sigmoid)
yhat = m(x3)
@test size(yhat) == (256,256,1,1) || @error "logits error"


# return_fetuares
m = espnet(3,2, alpha2=2, alpha3=3)   # implicit activation="prelu"
yhat = m(x3; return_features=true)
@test size(yhat.encoder.cat1) == (128,128,19,1) || @error "encoder.ct1 error"
@test size(yhat.encoder.cat2) == (64,64,131,1)  || @error "encoder.ct2 error"
@test size(yhat.encoder.cat3) == (32,32,256,1)  || @error "encoder.ct3 error"
@test size(yhat.logits)       == (256,256,2,1)  || @error "logits error"

m = espnet(3,2, activation=leakyrelu, alpha2=2, alpha3=3)
yhat = m(x3; return_features=true)
@test size(yhat.encoder.cat1) == (128,128,19,1) || @error "encoder.ct1 error"
@test size(yhat.encoder.cat2) == (64,64,131,1)  || @error "encoder.ct2 error"
@test size(yhat.encoder.cat3) == (32,32,256,1)  || @error "encoder.ct3 error"
@test size(yhat.logits)       == (256,256,2,1)  || @error "logits error"
