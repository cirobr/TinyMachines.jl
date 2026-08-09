# convolution + batch normalization
CB(ch_in, ch_out, activation) = 
    Chain(
        ConvK3(ch_in, ch_out, activation),
        ConvK3(ch_out, ch_out),
        BatchNorm(ch_out, activation),
)


# maxpooling + convolution + batch normalization
MCB(ch_in, ch_out, activation) = 
    Chain(
        MaxPool((2,2); stride=2),
        ConvK3(ch_in, ch_out, activation),
        ConvK3(ch_out, ch_out),
        BatchNorm(ch_out, activation),
)
