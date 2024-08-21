# add similar testing for p-relu
x = rand(Float32, (256,256,3,1))
m = ESPnet(3,2; activation=relu)
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = ESPnet(3,2; activation=relu, alpha2=5, alpha3=8)  # max alphas on article
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"
