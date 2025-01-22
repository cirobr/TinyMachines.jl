# convolutional block
CBlock(ch_in, ch_out, activation) = 
    Chain(ConvK3(ch_in, ch_out), x -> activation.(x),
          ConvK3(ch_out, ch_out), BatchNorm(ch_out), x -> activation.(x)
)


# maxpooling + convolutional block
MCBlock(ch_in, ch_out, activation) = 
    Chain(MaxPool((2,2); stride=2),
          ConvK3(ch_in, ch_out), x -> activation.(x),
          ConvK3(ch_out, ch_out), BatchNorm(ch_out), x -> activation.(x)
)
