struct PReLU
    vector::AbstractArray
end
@layer PReLU

function PReLU(ch::Int)
    vector = rand32(1,1,ch)
    return PReLU(vector)
end

function (m::PReLU)(x)
    return fpos.(x) .+ (m.vector .* fneg.(x))
end
