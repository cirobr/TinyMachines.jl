module TinyMachines


export UNet5, UNet4, UNet2
export MobileUNet
export ESPnet
# export prelu

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel,
             kaiming_normal,
             identity, relu, relu6, σ, sigmoid, softmax,
             @layer


# packages
const w1 = 1
const w2 = 2 * 2
const w3 = 3 * 3
const w4 = 4 * 4
const kf = 1.f-2
include("./functions/convolutions.jl")
include("./functions/unetblocks.jl")     # unet blocks
include("./functions/irblocks.jl")       # inverted residual blocks
include("./functions/espblocks.jl")      # efficient spatial pyramid blocks

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
