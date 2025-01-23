# convolutional block
CBlock(ch_in, ch_out, activation) = 
    Chain(ConvK3(ch_in, ch_out, activation),
          ConvK3(ch_out, ch_out), BatchNorm(ch_out, activation)
)


# maxpooling + convolutional block
MCBlock(ch_in, ch_out, activation) = 
    Chain(MaxPool((2,2); stride=2),
          ConvK3(ch_in, ch_out, activation),
          ConvK3(ch_out, ch_out), BatchNorm(ch_out, activation)
)


# up convolutional block
UpBlock(ch_in, ch_out, activation) = 
    ConvTranspK2(ch_in, ch_out, activation; stride=2)