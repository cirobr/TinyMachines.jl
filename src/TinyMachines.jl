module TinyMachines


export UNet, UNet5, UNet4, MobileUNet, ESPNet   # models
export unet, unet5, unet4, mobileunet, espnet   # constructors
export PReLU                                    # activation layer
export unet5H, UNet5H

import Flux
import Flux:
    Chain, SkipConnection, Parallel, Conv, ConvTranspose, DepthwiseConv,
    MaxPool, MeanPool, Upsample, 
    BatchNorm, Dropout,
    identity, relu, leakyrelu, relu6, σ, sigmoid, softmax,
    SamePad, kaiming_normal, rand32,
    @layer

include("functions/misc.jl")               # miscellaneous functions
include("functions/convlayers.jl")         # convolutional layers (custom conv + activation)
include("functions/activationlayers.jl")   # activation layers (prelu)
include("functions/unetblocks.jl")         # unet blocks
include("functions/irblocks.jl")           # inverted residual blocks
include("functions/espmodules.jl")         # esp modules

include("functions/wavelets.jl")

# models
const defaultChannels = [64, 128, 256, 512, 1024]

include("models/unet5.jl")
include("models/unet4.jl")
include("models/mobileunet.jl")
include("models/espnet.jl")

include("models/unet5-haar.jl")


end   # module
