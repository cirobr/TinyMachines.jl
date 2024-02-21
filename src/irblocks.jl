# inverted residual bottleneck block
function irbottleneck(ch_in::Int, ch_out::Int, stride::Int, expand_ratio::Int)
    if stride ∉ [1, 2]   return -1   end   # error
    ch_exp = ch_in * expand_ratio
    kgain  = kf * √(w3 * ch_in)

    return Chain(
        ConvK1(ch_in, ch_exp), BatchNorm(ch_exp, relu6),
        DepthwiseConv((3, 3), ch_exp => ch_exp, stride=stride, pad=1,
                      init=kaiming_normal(gain=kgain)),
        BatchNorm(ch_exp, relu6),
        ConvK1(ch_exp, ch_out)
    )
end


# stride 2 block
irblock2(ch_in, ch_out, expand_ratio=6) = 
irbottleneck(ch_in, ch_out, 2, expand_ratio)


# stride 1 block / skip connection
irblock1(ch_in, ch_out, expand_ratio=6) =
Parallel(+,
         irbottleneck(ch_in, ch_out, 1, expand_ratio),   # main branch
         ConvK1(ch_in, ch_out)                           # skip connection
)


# inverted residual block
function irblock(ch_in, ch_out; stride=2, expand_ratio=6)
    if stride == 1
        return irblock1(ch_in, ch_out, expand_ratio)
    elseif stride == 2
        return irblock2(ch_in, ch_out, expand_ratio)
    else
        return -1 # error
    end
end


# chain irblocks
function irblocks(ch_in, ch_out; n=1, stride=2, expand_ratio=6)
    model  =  irblock(ch_in, ch_out, stride=stride, expand_ratio=expand_ratio)
    modeln = [irblock1(ch_out, ch_out, expand_ratio) for i in 2:n]
    return Chain(model, modeln...)
end
