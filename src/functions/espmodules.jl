# input image downsampling
# const _pool_2x = MeanPool((3,3); stride=2, pad=1)
# const _pool_4x = Chain(MeanPool((3,3); stride=2, pad=1),
#                        MeanPool((3,3); stride=2, pad=1)
# )

# img_ds1(x) = _pool_2x(x)   # 2× downsampling of raw input (sample1)
# img_ds2(x) = _pool_4x(x)   # 4× downsampling of raw input (sample2)



# ESP downsampling
struct DownSampler
    c1::Conv
    bn1::BatchNorm
    d1::Conv
    d2::Conv
    d4::Conv
    d8::Conv
    d16::Conv
    bn::BatchNorm
    act::Any
end
@layer DownSampler

function DownSampler(ch_in::Int, ch_out::Int; activation = identity)
    if ch_out < 5   # fallback channels
        c1  = ConvK3(ch_in, ch_out; stride=2)
        bn1 = BatchNorm(ch_out)
        act = (activation == "prelu" ? PReLU(ch_out) : activation)
        return Chain(c1, bn1, act)
    end
    
    n  = ch_out ÷ 5
    n1 = ch_out - 4 * n
    
    c1  = ConvK3(ch_in, n; stride=2)          # strided 3x3 reducer (no pointwise)
    bn1 = BatchNorm(n)
    d1  = DilatedConvK3(n, n1; dilation=1)
    d2  = DilatedConvK3(n, n;  dilation=2)
    d4  = DilatedConvK3(n, n;  dilation=4)
    d8  = DilatedConvK3(n, n;  dilation=8)
    d16 = DilatedConvK3(n, n;  dilation=16)
    bn  = BatchNorm(ch_out)
    act = (activation == "prelu" ? PReLU(ch_out) : activation)

    return DownSampler(c1, bn1, d1, d2, d4, d8, d16, bn, act)
end

function (m::DownSampler)(x::AbstractArray)
    output1 = m.bn1( m.c1(x) )
    d1  = m.d1(output1)
    d2  = m.d2(output1)
    d4  = m.d4(output1)
    d8  = m.d8(output1)
    d16 = m.d16(output1)

    add1 = d2
    add2 = add1 + d4
    add3 = add2 + d8
    add4 = add3 + d16

    combine = cat(d1, add1, add2, add3, add4; dims=3)
    output = m.bn(combine)
    output = m.act(output)
    return output
end



# ESP module
# Used with add=true for encoder repeats, add=false in decoder.
struct ESPModule
    c1::Conv
    bn1::BatchNorm          # BN that belongs to the 1x1 pointwise reduce (matches original C)
    d1::Conv
    d2::Conv
    d4::Conv
    d8::Conv
    d16::Conv
    bn::BatchNorm
    act::Any
    add::Bool
end
@layer ESPModule

function ESPModule(ch_in::Int, ch_out::Int; add::Bool = true, activation = identity)
    if ch_out < 5   # fallback channels
        c   = ConvK1(ch_in, ch_out)
        bn1 = BatchNorm(ch_out)
        d   = ConvK3(ch_out, ch_out)
        bn  = BatchNorm(ch_out)
        act = (activation == "prelu" ? PReLU(ch_out) : activation)
        return Chain(c, bn1, d, bn, act)
    end

    n  = ch_out ÷ 5
    n1 = ch_out - 4 * n
    
    c1  = ConvK1(ch_in, n)
    bn1 = BatchNorm(n)                    # BN after 1x1 reduce (as in original C)
    d1  = DilatedConvK3(n, n1; dilation=1)
    d2  = DilatedConvK3(n, n;  dilation=2)
    d4  = DilatedConvK3(n, n;  dilation=4)
    d8  = DilatedConvK3(n, n;  dilation=8)
    d16 = DilatedConvK3(n, n; dilation=16)
    bn  = BatchNorm(ch_out)
    act = (activation == "prelu" ? PReLU(ch_out) : activation)
    
    return ESPModule(c1, bn1, d1, d2, d4, d8, d16, bn, act, add)
end

function (m::ESPModule)(x::AbstractArray)
    output1 = m.bn1( m.c1(x) )   # 1x1 pointwise reduce + its BN (matches original C)
    d1  = m.d1(output1)
    d2  = m.d2(output1)
    d4  = m.d4(output1)
    d8  = m.d8(output1)
    d16 = m.d16(output1)

    add1 = d2
    add2 = add1 + d4
    add3 = add2 + d8
    add4 = add3 + d16

    combine = cat(d1, add1, add2, add3, add4; dims=3)

    if m.add
        combine = x + combine
    end

    output = m.bn(combine)
    output = m.act(output)
    return output
end


# Backwards-compatible thin wrappers (so existing tests using ESP1/ESP4 still work)
function ESP1(ch_in::Int, ch_out::Int; activation = identity, stride::Int = 1)
    if stride == 2
        return DownSampler(ch_in, ch_out; activation=activation)
    else
        return ESPModule(ch_in, ch_out; add=true, activation=activation)
    end
end

function ESP4(ch_in::Int, ch_out::Int; activation = identity)
    return ESPModule(ch_in, ch_out; add=true, activation=activation)
end
