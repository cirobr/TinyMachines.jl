@info "espnet tests..."

m = ESPNet()
yhat = m(x3)
@test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"

m = espnet(3,2, alpha2=2, alpha3=3)   # implicit activation="prelu"
yhat = m(x3)
@test size(yhat[2])   == (128,128,19,1) || @error "size(yhat[2])   == $(size(yhat[2]))"
@test size(yhat[5])   == (64,64,131,1)  || @error "size(yhat[5])   == $(size(yhat[5]))"
@test size(yhat[8])   == (32,32,256,1)  || @error "size(yhat[8])   == $(size(yhat[8]))"
@test size(yhat[13])  == (64,64,4,1)    || @error "size(yhat[13])  == $(size(yhat[13]))"
@test size(yhat[15])  == (128,128,4,1)  || @error "size(yhat[15])  == $(size(yhat[15]))"
@test size(yhat[end]) == (256,256,2,1)  || @error "size(yhat[end]) == $(size(yhat[end]))"

m = espnet(3,2, activation=leakyrelu, alpha2=2, alpha3=3)
yhat = m(x3)
@test size(yhat[2])   == (128,128,19,1) || @error "size(yhat[2])   == $(size(yhat[2]))"
@test size(yhat[5])   == (64,64,131,1)  || @error "size(yhat[5])   == $(size(yhat[5]))"
@test size(yhat[8])   == (32,32,256,1)  || @error "size(yhat[8])   == $(size(yhat[8]))"
@test size(yhat[13])  == (64,64,4,1)    || @error "size(yhat[13])  == $(size(yhat[13]))"
@test size(yhat[15])  == (128,128,4,1)  || @error "size(yhat[15])  == $(size(yhat[15]))"
@test size(yhat[end]) == (256,256,2,1)  || @error "size(yhat[end]) == $(size(yhat[end]))"
