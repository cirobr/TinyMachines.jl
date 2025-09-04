@info "irblocks tests..."

m = tm.IRBlock1(3, 2, relu6, t=6)
yhat = m(x3)
@test size(yhat) == (256,256,2,1) || @error "size(yhat) == $(size(yhat))"

m = tm.IRBlock1(3, 3, relu6, t=6)
yhat = m(x3)
@test size(yhat) == (256,256,3,1) || @error "size(yhat) == $(size(yhat))"

m = tm.IRBlock2(3, 2, relu6, t=6)
yhat = m(x3)
@test size(yhat) == (128,128,2,1) || @error "size(yhat) == $(size(yhat))"
