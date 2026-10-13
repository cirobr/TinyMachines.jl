@info "espnet tests..."


# logits output
m = ESPNet()
yhat = m(x3)
@test size(yhat) == (256,256,2,1) || @error "wrapper error"

m = espnet(3,1)
yhat = m(x3)
@test size(yhat) == (256,256,1,1) || @error "logits 1ch error"

m = espnet(3,4)
yhat = m(x3)
@test size(yhat) == (256,256,4,1) || @error "logits 4ch error"

m = espnet(3,5)
yhat = m(x3)
@test size(yhat) == (256,256,5,1) || @error "logits 5ch error"


# TODO: encoder test