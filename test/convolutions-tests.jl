@test size(TinyMachines.ConvK1(1, 1)(x1)) == (256,256,1,1)
@test size(TinyMachines.ConvK1(3, 1)(x3)) == (256,256,1,1)

@test size(TinyMachines.ConvK2(1, 1)(x1)) == (128,128,1,1)
@test size(TinyMachines.ConvK2(3, 1)(x3)) == (128,128,1,1)

@test size(TinyMachines.ConvK3(1, 1)(x1)) == (256,256,1,1)
@test size(TinyMachines.ConvK3(3, 1)(x3)) == (256,256,1,1)
@test size(TinyMachines.ConvK3(1, 1; stride=2)(x1)) == (128,128,1,1)
@test size(TinyMachines.ConvK3(3, 1; stride=2)(x3)) == (128,128,1,1)

@test size(TinyMachines.UpConvK2(1, 1)(x1)) == (512,512,1,1)
@test size(TinyMachines.UpConvK2(3, 1)(x3)) == (512,512,1,1)

@test size(TinyMachines.ConvTranspK2(1, 1)(x1)) == (512,512,1,1)
@test size(TinyMachines.ConvTranspK2(3, 1)(x3)) == (512,512,1,1)

@test size(TinyMachines.ConvTranspK4(1,1)(x1)) == (512,512,1,1)
@test size(TinyMachines.ConvTranspK4(3,1)(x3)) == (512,512,1,1)

@test size(TinyMachines.MaxPoolK2(x1)) == (128,128,1,1)
@test size(TinyMachines.MaxPoolK2(x3)) == (128,128,3,1)
