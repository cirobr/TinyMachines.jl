function PWConv(ksize, ch_in::Int, ch_out::Int, activation=identity; K::Int, stride=1)
    d = ch_in / K |> Int
    if !isinteger(d)   error("ch_in must be divisible by K.")   end

    pad = (ksize - 1)/2 |> Int
    w = ksize * ksize
    kgain = kf * √(w * ch_in)

    return Conv((1,1), ch_in => ch_out, activation; stride=stride, pad=pad,
                bias=false,
                groups=d,
                init=kaiming_normal(gain=kgain)
                )
end

PWConv(3, 256, 256, relu, K=1, stride=1)