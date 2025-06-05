module TinyMachines


export UNet5, UNet4, MobileUNet, ESPNet   # models
export unet5, unet4 #, mobileunet, espnet   # constructors
export ConvPReLU                          # special layers

import Flux
import Flux: Chain, SkipConnection, Parallel, Conv, ConvTranspose, DepthwiseConv,
             MaxPool, Upsample, 
             BatchNorm, Dropout,
             identity, relu, relu6, σ, sigmoid, softmax,
             SamePad, kaiming_normal, rand32,
             @layer


# functions
const w1 = 1
const w2 = 2 * 2
const w3 = 3 * 3
const w4 = 4 * 4
const kf = 1.f-2

include("./functions/convlayers.jl")         # convolutional layers (custom conv + activation)
include("./functions/activationlayers.jl")   # activation layers (prelu)
include("./functions/unetblocks.jl")         # unet blocks
include("./functions/irblocks.jl")           # inverted residual blocks
include("./functions/espblocks.jl")          # esp blocks

# models
const defaultChannels = [64, 128, 256, 512, 1024]

include("./models/unet5.jl")
include("./models/unet4.jl")
include("./models/mobileunet.jl")
include("./models/espnet.jl")


end   # module
