# add similar testing for p-relu
x = rand(Float32, (256,256,3,1))
m = ESPnet(3,2; activation=relu)
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = ESPnet(3,2; activation=relu, alpha2=5, alpha3=8)  # max alphas on article
yhat = m(x)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = ESPnet(3,2; activation=relu, verbose=true)
yhat = m(x)
s = size.(yhat)
@test s[1] == (256,256,2,1) || @error "s[1] == $s[1]"
@test s[2] == (256,256,2,1) || @error "s[2] == $s[2]"
