Random.seed!(1234)
x = randn(Float32, (3,3,1,1))
yhat = ConvPReLU(1)(x)

@test x[1,1,1,1] != yhat[1,1,1,1] || error("x is negative, yhat should be different from x.")
@test x[2,1,1,1] == yhat[2,1,1,1] || error("x is positive, yhat should be equal to x.")
@test sign.(x)   == sign.(yhat)   || error("x and yhat should have the same sign for all elements.")
