# Test the ESPmodule
esp = ESPmodule(3, 12; K=4, add=false)
y = esp(x3)
@test size(y) == (256,256,12,1)


ch_in, ch_out = 12, 12
esp = ESPmodule(ch_in, ch_out; K=4, add=true)
xesp = randn(Float32, (256, 256, ch_in, ch_out))
y = esp(xesp)
@test size(y) == (256,256,12,12)


# Test the ESPNet
espnet = ESPNet(3, 12; K=4)
y = espnet(x3)
@test size(y) == (256,256,12,1)
