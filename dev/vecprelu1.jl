struct vecprelu1
    α::Vector{Float32}  # trainable parameters, one per channel
end

# Make vecprelu1 a Flux "layer" with trainable parameters
Flux.@layer vecprelu1

function vecprelu1(ch_in::Int)
    α = rand(Float32, ch_in)  # random initialization in [0, 1)
    return vecprelu1(α)
end

# Callable: input x is (C, ...) where C = ch_in
function (m::vecprelu1)(x)
    α_broadcast = reshape(m.α, 1, 1, :, 1)
    return max.(x, 0) .+ α_broadcast .* min.(x, 0)
end
