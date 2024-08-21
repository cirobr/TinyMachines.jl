x = rand(Float32, (256,256,3,1))
m = tm.ESPModule1(3, 2, relu; stride=1, add=false)
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule1(3, 3, relu; stride=1, add=true)
yhat = m(x)
@test size(yhat) == (256,256,3,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule1(3, 2, relu; stride=2, add=false)
yhat = m(x)
@test size(yhat) == (128,128,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule4(3, 4, relu)
yhat = m(x)
@test size(yhat) == (256,256,4,1) || @error "size(yhat) == $(size(yhat))"

x = rand(Float32, (256,256,8,1))   # C must be divisible by 4
m = tm.ESPModule4_alpha(8, relu; alpha=3)
yhat = m(x)
@test size(yhat) == (256,256,8,1) || @error "size(yhat) == $(size(yhat))"

# add similar testing for p-relu
x = rand(Float32, (256,256,3,1))
m = ESPnet(3,2; activation=relu)
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = ESPnet(3,2; activation=relu, alpha2=5, alpha3=8)  # max alphas on article
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"
