# inverted residual bottleneck block
function irbottleneck(ch_in::Int, ch_out::Int, stride::Int, expand_ratio::Int)
    ch_exp = ch_in * expand_ratio
    kgain  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp), BatchNorm(ch_exp, relu6),
        DepthwiseConv((3, 3), ch_exp => ch_exp, identity;
                      stride=stride,
                      pad=1,
                      bias=true,
                      dilation=1,
                      init=kaiming_normal(gain=kgain)
        ),
        BatchNorm(ch_exp, relu6),
        ConvK1(ch_exp, ch_out, identity)
    )
end


# irbottleneck stride 2
irb2(ch_in, ch_out, expand_ratio) = 
irbottleneck(ch_in, ch_out, 2, expand_ratio)


# irbottleneck stride 1 / skip connection
irb1(ch_in, ch_out, expand_ratio) =
Parallel(+,
         irbottleneck(ch_in, ch_out, 1, expand_ratio),   # main branch
         ConvK1(ch_in, ch_out, identity)                 # skip connection
)


function irblock2(ch_in, ch_out; n=1, expand_ratio=6)
    model_in    =  irb2(ch_in, ch_out, expand_ratio)
    model_chain = [irb1(ch_out, ch_out, expand_ratio) for i in 2:n]

    return n == 1 ? model_in : Chain(model_in, model_chain...)
end


function irblock1(ch_in, ch_out; n=1, expand_ratio=6)
    model_in    =  irb1(ch_in, ch_out, expand_ratio)
    model_chain = [irb1(ch_out, ch_out, expand_ratio) for i in 2:n]

    return n == 1 ? model_in : Chain(model_in, model_chain...)
end
