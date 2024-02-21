module TinyMachines


export UNet5, UNet4, UNet2
export MobUNet
export ESPNet

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout,
             Scale, DepthwiseConv, Parallel,
             identity, relu, σ, sigmoid, softmax, relu6,
             @functor, kaiming_normal
include("./convolutions.jl")
include("./irblocks.jl")   # inverted residual blocks

# unets
const defaultChannels = [32, 64, 128, 256, 512]
include("./unet5.jl")
include("./unet4.jl")
include("./unet2.jl")

# mobile unet
include("./mobileunet.jl")


end   # module
