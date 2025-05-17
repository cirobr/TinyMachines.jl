m = tm.ESPBlock1(3, 2; stride=1, add=false)
yhat = m(x3)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPBlock1(3, 3; stride=1, add=true)
yhat = m(x3)
@test size(yhat) == (256,256,3,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPBlock1(3, 2; stride=2, add=false)
yhat = m(x3)
@test size(yhat) == (128,128,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPBlock4(3, 4)
yhat = m(x3)
@test size(yhat) == (256,256,4,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPBlock4_alpha(8; alpha=3)
yhat = m(x8)
@test size(yhat) == (256,256,8,1) || @error "size(yhat) == $(size(yhat))"
