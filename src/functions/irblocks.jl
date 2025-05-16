# inverted bottleneck residual block
function BRBlock(ch_in::Int, ch_out::Int, activation::Function, stride::Int, expansion_factor::Int)
    ch_exp = ch_in * expansion_factor
    gn  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp), BatchNorm(ch_exp, activation),
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
function IRBlock1(ch_in::Int, ch_out::Int, activation::Function, expansion_factor::Int)
    skipconn = ch_in == ch_out ? identity : ConvK1(ch_in, ch_out)

    return Parallel(
        +,
        BRBlock(ch_in, ch_out, activation, 1, expansion_factor),
        skipconn
    )
end


# irblock stride 2
function IRBlock2(ch_in::Int, ch_out::Int, activation::Function, expansion_factor::Int)
    return BRBlock(ch_in, ch_out, activation, 2, expansion_factor)
end


# bottleneck block   1,6,1
function BBlock(ch_in::Int, ch_out::Int, activation::Function; stride::Int, expansion_factor::Int, n::Int)
    @assert stride ∈ [1, 2] || error("stride must be 1 or 2")
    model_in = (stride == 1) ?
                IRBlock1(ch_in, ch_out, activation, expansion_factor) :   # stride 1
                IRBlock2(ch_in, ch_out, activation, expansion_factor)     # stride 2
    model_chain = [IRBlock1(ch_out, ch_out, activation, expansion_factor) for _ in 1:n-1]

    return (n == 1) ? model_in : Chain(model_in, model_chain...)
end
