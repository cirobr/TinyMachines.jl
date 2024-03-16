struct model_prelu
    alpha::Vector{Float32}
end
Flux.@layer model_prelu

fn_prelu(x, α) = x>0 ? x : α * x

function (m::model_prelu)(x)
    return fn_prelu.(x, m.alpha[1])
end

prelu=model_prelu([0.f0])
