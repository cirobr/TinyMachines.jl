# convolutional block
cb(ch_in, ch_out, activation)  = Chain(ConvK3(ch_in, ch_out, activation),
                                       ConvK3(ch_out, ch_out), BatchNorm(ch_out, activation)
)

# maxpooling + convolutional block
mcb(ch_in, ch_out, activation) = Chain(MaxPool((2,2); stride=2),
                                       cb(ch_in, ch_out, activation)
)



"""
Flux layers accept a single input and return a single output.
The `UpBlock` struct is a wrapper around the `ConvTranspK2` layer.
Its objective is to create a layer with two inputs and a single output.
"""
# struct UpBlock
#     u
# end
# @layer UpBlock trainable=(u)

# function UpBlock(ch_in::Int, ch_out::Int, activation)
#     return UpBlock( ConvTranspK2(ch_in, ch_out, activation; stride=2) )
# end

# function (m::UpBlock)(dwn, skip)
#     x = m.u(dwn)
#     return cat(x, skip, dims=3)
# end

# cat4 = m.upc[:u4](enc5, enc4)
# cat3 = m.upc[:u3](dec4, enc3)
# cat2 = m.upc[:u2](dec3, enc2)
# cat1 = m.upc[:u1](dec2, enc1)
