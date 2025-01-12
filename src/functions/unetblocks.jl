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



### up block
struct UpBlock
    u
end
@layer UpBlock trainable=(u)

function UpBlock(ch_in::Int, ch_out::Int, act)
    return UpBlock( ConvTranspK2(ch_in, ch_out, act; stride=2) )
end

function (m::UpBlock)(x, enc)
    x = m.u(x)
    return cat(x, enc, dims=3)
end