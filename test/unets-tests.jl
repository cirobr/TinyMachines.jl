# UNet2
modelcpu = UNet2()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet2(1, 1; activation=Flux.leakyrelu, alpha=0.5, verbose=true)
yhat  = modelcpu(x1)
@test size.((yhat)[2]) == [(256, 256, 16, 1),
                           (128, 128, 32, 1),
                           (256, 256, 16, 1),
                           (256, 256, 1, 1)]


# UNet4
modelcpu = UNet4()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet4(1, 1; activation=Flux.leakyrelu, alpha=0.5, verbose=true)
yhat  = modelcpu(x1)
@test size.((yhat)[2]) == [(256, 256, 16, 1),
                           (128, 128, 32, 1),
                           (64, 64, 64, 1),
                           (32, 32, 128, 1),
                           (64, 64, 64, 1),
                           (128, 128, 32, 1),
                           (256, 256, 16, 1),
                           (256, 256, 1, 1)]


# UNet5
modelcpu = UNet5()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 1, 1)

modelcpu = UNet5(1, 1; activation=Flux.leakyrelu, alpha=0.5, verbose=true)
yhat  = modelcpu(x1)
@test size.((yhat)[2]) == [(256, 256, 16, 1),
                           (128, 128, 32, 1),
                           (64, 64, 64, 1),
                           (32, 32, 128, 1),
                           (16, 16, 256, 1),
                           (32, 32, 128, 1),
                           (64, 64, 64, 1),
                           (128, 128, 32, 1),
                           (256, 256, 16, 1),
                           (256, 256, 1, 1)]
