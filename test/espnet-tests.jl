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


# encoder sub-chain (Flux.activations on encoder)
# stages 1+2 return (fused_feature, original_x) tuples to thread image for fusion ds;
# the fused_feature is the pre-dropout value used for bridges (except stage3)
m = espnet(3, 3)
fs = Flux.activations(m.encoder, x3)
@test size(fs[1][1]) == (128,128,19,1) || @error "esp e1 fused"
@test size(fs[2][1]) == (64,64,131,1) || @error "esp e2 fused"
@test size(fs[3]) == (32,32,256,1) || @error "esp e3 fused"
yhat = m(x3)
@test size(yhat) == (256,256,3,1) || @error "esp logits ch3 error"
