# inverted residual bottleneck block
function bottleneck_residual_block(ch_in::Int, ch_out::Int, stride::Int, expansion_factor::Int)
    ch_exp = ch_in * expansion_factor
    kgain  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp), BatchNorm(ch_exp, relu6),
        DepthwiseConv((3, 3), ch_exp => ch_exp;
                      stride=stride,
                      pad=SamePad(),
                      bias=true,
                      dilation=1,
                      init=kaiming_normal(gain=kgain)
        ), BatchNorm(ch_exp, relu6),
        ConvK1(ch_exp, ch_out)
    )
end


# # debug version (missing skip connection)
# irblock1(ch_in, ch_out, expansion_factor) =
#         bottleneck_residual_block(ch_in, ch_out, 1, expansion_factor)

# ir block stride 1
function irblock1(ch_in, ch_out, expansion_factor)
    skipconn = ch_in == ch_out ? identity : ConvK1(ch_in, ch_out)

    return Parallel(
        +,
        bottleneck_residual_block(ch_in, ch_out, 1, expansion_factor),
        skipconn
    )
end


# irblock stride 2
function irblock2(ch_in, ch_out, expansion_factor)
    return bottleneck_residual_block(ch_in, ch_out, 2, expansion_factor)
end


function n_irblock1(ch_in, ch_out; n=1, expansion_factor=6)
    model_in    =  irblock1(ch_in, ch_out, expansion_factor)
    model_chain = [irblock1(ch_out, ch_out, expansion_factor) for i in 1:n-1]

    return n == 1 ? model_in : Chain(model_in, model_chain...)
end


function n_irblock2(ch_in, ch_out; n=1, expansion_factor=6)
    model_in    =  irblock2(ch_in, ch_out, expansion_factor)
    model_chain = [irblock1(ch_out, ch_out, expansion_factor) for i in 1:n-1]
    
    return n == 1 ? model_in : Chain(model_in, model_chain...)
end
