fn_prelu(x, alpha) = x>0 ? x : alpha * x

struct model_prelu
    alpha::Vector{Float32}
end

function (m::model_prelu)(x)
    return fn_prelu.(x, m.alpha[1])
end
Flux.@functor model_prelu

prelu=model_prelu([0.f0])
