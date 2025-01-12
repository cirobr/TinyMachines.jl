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
struct UpBlock
    u
end
@layer UpBlock trainable=(u)

function UpBlock(ch_in::Int, ch_out::Int, activation=identity)
    return UpBlock( ConvTranspK2(ch_in, ch_out, activation; stride=2) )
end

function (m::UpBlock)(dwn, skip)
    x_up = m.u(dwn)
    return cat(x_up, skip, dims=3)
end
