@test size(MobileUNet(1,1)(x1)) == (256,256,1,1)
@test size(MobileUNet(3,1)(x3)) == (256,256,1,1)
