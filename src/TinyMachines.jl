module TinyMachines


export UNet5, UNet4, UNet2

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, Scale,
             identity, relu, σ, sigmoid, softmax,
             @functor, kaiming_normal
include("./activations.jl")
include("./convolutions.jl")

# unets
const defaultChannels = [32, 64, 128, 256, 512]
include("./unet5.jl")
include("./unet4.jl")
include("./unet2.jl")

# espnet
include("./espnet.jl")


end   # module
