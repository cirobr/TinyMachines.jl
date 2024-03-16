module TinyMachines


export UNet5, UNet4, UNet2
export MobileUNet
export ESPNet, ESPmodule, ESPmoduleK1, prelu

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel,
             identity, relu, σ, sigmoid, softmax, relu6,
             @layer, kaiming_normal

# packages
include("./pkgs/convolutions.jl")
include("./pkgs/irblocks.jl")             # inverted residual blocks
include("./pkgs/prelu.jl")                # PReLU
include("./espnet-src/espmodule-2.jl")    # espmodule_k preferred


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
