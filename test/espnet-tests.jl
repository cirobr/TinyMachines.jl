x = rand(Float32, (256,256,3,4))
m = ESPmodule(3, 4; K=4)
yhat = m(x)
@test size(yhat) == (256,256,4,4) || @error "size(yhat) == $(size(yhat))"

x = rand(Float32, (256,256,4,4))
m = ESPmodule(4, 4; K=4, add=true)
yhat = m(x)
@test size(yhat) == (256,256,4,4) || @error "size(yhat) == $(size(yhat))"

x = rand(Float32, (256,256,3,1))
m = ESPNet(3,1; activation=relu, K=4)
yhat = m(x)
@test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"

x = rand(Float32, (256,256,3,1))
m = ESPNet(3,1; activation=prelu, K=4)
yhat = m(x)
@test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"
