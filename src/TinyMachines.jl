module TinyMachines


export UNet5, UNet4, UNet2
export MobUNet
export ESPNet, ESPmodule

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             Scale, DepthwiseConv, Parallel,
             identity, relu, σ, sigmoid, softmax, relu6,
             @functor, kaiming_normal
include("./pkgs/activations.jl")
include("./pkgs/convolutions.jl")
include("./pkgs/irblocks.jl")   # inverted residual blocks

# unets
const defaultChannels = [32, 64, 128, 256, 512]
include("./models/unet5.jl")
include("./models/unet4.jl")
include("./models/unet2.jl")

# mobile unet
include("./models/mobileunet.jl")

# espnet
include("./espmodule.jl")
include("./espnet.jl")


end   # module
