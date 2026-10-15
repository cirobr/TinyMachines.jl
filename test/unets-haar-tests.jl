@info "unets-haar tests..."


# UNet5
modelcpu = tm.UNet5_v1a()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 2, 1) || @error "wrapper error"

modelcpu = tm.UNet5_v1b()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 2, 1) || @error "wrapper error"
