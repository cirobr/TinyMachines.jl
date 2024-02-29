module TinyMachines


export UNet5, UNet4, UNet2
export MobUNet
export ESPNet, ESPmoduleK4, ESPmoduleK1

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel,
             Scale,
             identity, relu, σ, sigmoid, softmax, relu6,
             @functor, kaiming_normal

# packages
include("./pkgs/convolutions.jl")
include("./pkgs/irblocks.jl")       # inverted residual blocks
include("./espnet-src/activations.jl")    # PReLU
include("./espnet-src/espmodule_k.jl")    # espmodule_k preferred


# unets
const defaultChannels = [64, 128, 256, 512, 1024]
include("./models/unet5.jl")
include("./models/unet4.jl")
include("./models/unet2.jl")

# mobile unet
include("./models/mobileunet.jl")

# espnet
include("./espnet-src/espnet.jl")


end   # module
