module TinyMachines


export UNet5, UNet4, MobileUNet, ESPnet

import Flux
import Flux: Chain, SkipConnection, Conv, MaxPool, Upsample, ConvTranspose, BatchNorm, Dropout, SamePad,
             DepthwiseConv, Parallel,
             identity, relu, relu6, σ, sigmoid, softmax,
             kaiming_normal, rand32,
             @layer


# packages
const w1 = 1
const w2 = 2 * 2
const w3 = 3 * 3
const w4 = 4 * 4
const kf = 1.f-2

include("./functions/convlayers.jl")   # convolutional layers (custom conv + activation)
include("./functions/activationlayers.jl") # activation layers (prelu)
include("./functions/unetblocks.jl")   # unet blocks
include("./functions/irblocks.jl")     # inverted residual blocks
# include("./functions/espblocks-activk1.jl")
# include("./functions/espblocks-convprelu.jl")
include("./functions/espblocks-prelulayer.jl")

# unets
const defaultChannels = [64, 128, 256, 512, 1024]

include("./models/unet5.jl")
include("./models/unet4.jl")

# mobile unet
include("./models/mobileunet.jl")

# espnet
# include("./models/espnet-activk1.jl")
# include("./models/espnet-convprelu.jl")
include("./models/espnet-prelulayer.jl")


end   # module
