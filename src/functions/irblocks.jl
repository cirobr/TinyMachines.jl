# inverted bottleneck residual block
function BRBlock(ch_in::Int, ch_out::Int,   # input/output channels
                 activation::Function;      # activation function
                 stride::Int,               # stride
                 t::Int                     # internal channel expansion factor
)
    ch_exp = ch_in * t
    gn  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp),
        BatchNorm(ch_exp, activation),
        DepthwiseConv((3, 3), ch_exp => ch_exp;
                        stride=stride,
                        pad=SamePad(),
                        bias=true,
                        # dilation=1,
                        init=kaiming_normal(gain=gn)
        ),
        BatchNorm(ch_exp, activation),
        ConvK1(ch_exp, ch_out)
    )
end


# irblock stride 1
function IRBlock1(ch_in::Int, ch_out::Int,   # input/output channels
                  activation::Function;      # activation function
                  t::Int                     # BRBlock internal channel expansion factor
)
    skipconn = ch_in == ch_out ? identity : ConvK1(ch_in, ch_out)
    return Parallel(
        +,
        BRBlock(ch_in, ch_out, activation; stride=1, t=t),
        skipconn
    )
end


# irblock stride 2
function IRBlock2(ch_in::Int, ch_out::Int,   # input/output channels
                  activation::Function;      # activation function
                  t::Int                     # BRBlock internal channel expansion factor
)
    return BRBlock(ch_in, ch_out, activation; stride=2, t=t)
end
