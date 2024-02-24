# Test the ESPmodule
esp = ESPmodule(3, 10; K=5, add=false)
y = esp(x3)
@test size(y) == (256,256,10,1)

ch_in, ch_out = 10, 10
esp = ESPmodule(ch_in, ch_out; K=5, add=true)
xesp = randn(Float32, (256, 256, ch_in, 1))
y = esp(xesp)
@test size(y) == (256,256,10,1)


# Test the ESPNet
espnet = ESPNet(3, 12; K=4)
y = espnet(x3)
@test size(y) == (256,256,12,1)
