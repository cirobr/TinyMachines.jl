# inverted residual bottleneck block
function irbottleneck(ch_in::Int, ch_out::Int, stride::Int, expansion_factor::Int)
    ch_exp = ch_in * expansion_factor
    kgain  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp, relu6),
        DepthwiseConv((3, 3), ch_exp => ch_exp, relu6;
                      stride=stride,
                      pad=SamePad(),
                      bias=true,
                      dilation=1,
                      init=kaiming_normal(gain=kgain)
        ),
        ConvK1(ch_exp, ch_out, identity)
    )
end


### irblock stride 2 (skip connection)

# debug version (missing skip connection)
# irblock2(ch_in, ch_out, expansion_factor) =
#         irbottleneck(ch_in, ch_out, 2, expansion_factor)

# with skip connection
irblock2(ch_in, ch_out, expansion_factor) =
    Parallel(
        +,
        irbottleneck(ch_in, ch_out, 2, expansion_factor),
        Conv((1,1), ch_in => ch_out, identity; stride=2, bias=false)   # not clear from the article
    )


# irblock stride 1
irblock1(ch_in, ch_out, expansion_factor) =
        irbottleneck(ch_in, ch_out, 1, expansion_factor)


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
