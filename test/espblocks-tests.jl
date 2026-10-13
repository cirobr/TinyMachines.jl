@info "esp blocks tests..."

# DownSampler
m = tm.DownSampler(3,1; activation="prelu")
yhat = m(x3)
@test size(yhat) == (128,128,1,1) || @error "size(yhat) == $(size(yhat))"

m = tm.DownSampler(3,4; activation="prelu")
yhat = m(x3)
@test size(yhat) == (128,128,4,1) || @error "size(yhat) == $(size(yhat))"

m = tm.DownSampler(3,5; activation="prelu")
yhat = m(x3)
@test size(yhat) == (128,128,5,1) || @error "size(yhat) == $(size(yhat))"


# ESP Module
m = tm.ESPModule(3,1; add=false, activation="prelu")
yhat = m(x3)
@test size(yhat) == (256,256,1,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule(3,4; add=false, activation="prelu")
yhat = m(x3)
@test size(yhat) == (256,256,4,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule(3,5; add=false, activation="prelu")
yhat = m(x3)
@test size(yhat) == (256,256,5,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule(3,3; add=true, activation="prelu")
yhat = m(x3)
@test size(yhat) == (256,256,3,1) || @error "size(yhat) == $(size(yhat))"

m = tm.ESPModule(5,5; add=true, activation="prelu")
yhat = m(x5)
@test size(yhat) == (256,256,5,1) || @error "size(yhat) == $(size(yhat))"
