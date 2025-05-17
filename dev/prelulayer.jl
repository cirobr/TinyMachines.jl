# PReLU layer
oftf(x, y) = oftype(float(x), y)

struct PReLUlayer
    a::AbstractArray
    # ch_in::Int
end
@layer PReLUlayer #trainable=(a)

function PReLUlayer(ch_in::Int)
    a = rand(1,1,ch_in,1)
    # a = rand(ch_in)

    return PReLUlayer(a)
end

function (m::PReLUlayer)(x)
    return @. max(x, 0) + oftf(x, m.a) * min(x, 0)
end
