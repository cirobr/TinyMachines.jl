# Dummy data for testing
Xs = randn(Float32, 64,64,3,10)
ys = rand(Bool, 64,64,2,10)
data = Flux.DataLoader((Xs, ys))

# Instantiate models
models = [
    ESPNet(3,2),
    MobileUNet(3,2),
    UNet4(3,2),
    UNet5(3,2),
]

# Test training
for model in models
    loss(model,x,y) = Flux.crossentropy(model(x), y; dims=3)
    opt = Flux.Adam()
    opt_state = Flux.setup(opt, model)
    Flux.train!(loss, model, data, opt_state)
    l = loss(model, Xs, ys)
    @test !isnan(l)       || error("$model : Loss is NaN")
    @test !isinf(l)       || error("$model : Loss is Inf")
    @test isa(l, Float32) || error("$model : Loss is not Float32")
end