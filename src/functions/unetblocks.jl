# convolutional block
cb(ch_in, ch_out, activation)  = Chain(ConvK3(ch_in, ch_out, activation),
                                       ConvK3(ch_out, ch_out), BatchNorm(ch_out, activation)
)

# maxpooling + convolutional block
mcb(ch_in, ch_out, activation) = Chain(MaxPool((2,2); stride=2),
                                       ConvK3(ch_in, ch_out, activation),
                                       ConvK3(ch_out, ch_out), BatchNorm(ch_out, activation)
)
