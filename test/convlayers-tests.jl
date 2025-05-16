@test size(tm.ConvK1(3, 1)(x3)) == (256,256,1,1)

@test size(tm.ConvK2(3, 1)(x3)) == (128,128,1,1)

@test size(tm.ConvK3(3, 1)(x3)) == (256,256,1,1)
@test size(tm.ConvK3(3, 1; stride=2)(x3)) == (128,128,1,1)

@test size(tm.UpConvK2(3, 1)(x3)) == (512,512,1,1)

@test size(tm.ConvTranspK2(3, 1)(x3)) == (256,256,1,1)
@test size(tm.ConvTranspK2(3, 1; stride=2)(x3)) == (512,512,1,1)

@test size(tm.ConvTranspK4(3, 1)(x3)) == (512,512,1,1)

# @test size(tm.MaxPoolK2(x3)) == (128,128,3,1)

@test size(tm.DilatedConvK3(3,1; stride=1, dilation=1)(x3)) == (256,256,1,1)
@test size(tm.DilatedConvK3(3,1; stride=1, dilation=2)(x3)) == (256,256,1,1)
@test size(tm.DilatedConvK3(3,1; stride=2, dilation=1)(x3)) == (128,128,1,1)
@test size(tm.DilatedConvK3(3,1; stride=2, dilation=2)(x3)) == (128,128,1,1)
