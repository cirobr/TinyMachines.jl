module TinyMachines


export UNet5, UNet4, UNet2
export MobUNet
export ESPNet, ESPmodule, prelu1, prelu2

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel, Scale,
             identity, relu, σ, sigmoid, softmax, relu6,
             gpu,
             @functor, kaiming_normal

# packages
include("./pkgs/activations.jl")    # PReLU
include("./pkgs/convolutions.jl")
include("./pkgs/irblocks.jl")       # inverted residual blocks
include("./pkgs/espmodule.jl")      # efficient spatial pyramid module

# unets
const defaultChannels = [64, 128, 256, 512, 1024]
include("./models/unet5.jl")
include("./models/unet4.jl")
include("./models/unet2.jl")

# mobile unet
include("./models/mobileunet.jl")

# espnet
include("./models/espnet.jl")


end   # module
