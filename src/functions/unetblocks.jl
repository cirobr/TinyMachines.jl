# convolutional block
cb(ch_in, ch_out, act)  = Chain(ConvK3(ch_in, ch_out, act),
                                ConvK3(ch_out, ch_out), BatchNorm(ch_out, act)
)

# maxpooling + convolutional block
mcb(ch_in, ch_out, act) = Chain(MaxPoolK2,
                                ConvK3(ch_in, ch_out, act),
                                ConvK3(ch_out, ch_out), BatchNorm(ch_out, act)
)

# up convolutional block
ub(ch_in, ch_out, act)  = ConvTranspK2(ch_in, ch_out, act; stride=2)
