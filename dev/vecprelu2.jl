struct vecprelu2
    α::Array{Float32,4}  # trainable parameters, one per channel
end

# Make vecprelu2 a Flux "layer" with trainable parameters
Flux.@layer vecprelu2

function vecprelu2(ch_in::Int)
    α = rand(Float32, (1,1,ch_in,1))  # random initialization in [0, 1)
    return vecprelu2(α)
end

# Callable: input x is (C, ...) where C = ch_in
function (m::vecprelu2)(x)
    return max.(x, 0) .+ m.α .* min.(x, 0)
end
