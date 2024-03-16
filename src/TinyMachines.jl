module TinyMachines


export UNet5, UNet4, UNet2
export MobileUNet
export ESPNet, ESPmoduleK4, ESPmoduleK1, prelu

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel,
             identity, relu, σ, sigmoid, softmax, relu6,
             @layer, kaiming_normal

# packages
include("./functions/convolutions.jl")
include("./functions/irblocks.jl")       # inverted residual blocks
include("./functions/prelu.jl")          # PReLU
include("./functions/espmodule.jl")    # 4-levels ESP module


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
