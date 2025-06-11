@info "espnet tests..."

m = ESPNet()
yhat = m(x3)
@test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"

m = espnet(3,2, alpha2=2, alpha3=3)  # min alphas on article
yhat = m(x3)
@test size(yhat[1])  == (128,128,3,1)  || @error "yhat == $(size(yhat[1]))"
@test size(yhat[3])  == (128,128,19,1) || @error "size(yhat[2][3]) == $(size(yhat[2][3]))"
@test size(yhat[7])  == (64,64,131,1)  || @error "size(yhat[2][7]) == $(size(yhat[2][7]))"
@test size(yhat[10]) == (32,32,256,1)  || @error "size(yhat[2][10]) == $(size(yhat[2][10]))"
@test size(yhat[11]) == (128,128,2,1)  || @error "size(yhat[2][11]) == $(size(yhat[2][11]))"
@test size(yhat[12]) == (64,64,2,1)    || @error "size(yhat[2][12]) == $(size(yhat[2][12]))"
@test size(yhat[13]) == (32,32,2,1)    || @error "size(yhat[2][13]) == $(size(yhat[2][13]))"
@test size(yhat[15]) == (64,64,4,1)    || @error "size(yhat[2][15]) == $(size(yhat[2][15]))"
@test size(yhat[17]) == (128,128,4,1)  || @error "size(yhat[2][17]) == $(size(yhat[2][17]))"
@test size(yhat[18]) == (256,256,2,1)  || @error "size(yhat[2][18]) == $(size(yhat[2][18]))"
