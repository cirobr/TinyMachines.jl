modelcpu = UNet4dw()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)
