x = rand(Float32, (256,256,3,1))
m = tm.ESPBlock1(3, 2, relu; stride=1, add=false)
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPBlock1(3, 3, relu; stride=1, add=true)
yhat = m(x)
@test size(yhat) == (256,256,3,1) || @error "size(yhat) == $(size(yhat))"


m = tm.ESPBlock1(3, 2, relu; stride=2, add=false)
yhat = m(x)
@test size(yhat) == (128,128,2,1) || @error "size(yhat) == $(size(yhat))"


m = tm.ESPBlock4(3, 4, relu)
yhat = m(x)
@test size(yhat) == (256,256,4,1) || @error "size(yhat) == $(size(yhat))"


x = rand(Float32, (256,256,8,1))   # C must be divisible by 4
m = tm.ESPBlock4_alpha(8, relu; alpha=3)
yhat = m(x)
@test size(yhat) == (256,256,8,1) || @error "size(yhat) == $(size(yhat))"
