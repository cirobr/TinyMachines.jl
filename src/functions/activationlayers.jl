struct PReLU
    weights::AbstractArray
end
@layer PReLU

function PReLU(ch::Int)
    weights = rand32(1,1,ch)
    return PReLU(weights)
end

function (m::PReLU)(x)
    return fpos.(x) .+ (m.weights .* fneg.(x))
end
