"""
Distributed computing with multiple GPUs
wpd defines number of workers per device (gpu)
# https://cuda.juliagpu.org/stable/usage/multigpu/

Author: cirobr@GitHub
Date: 02-Aug-2024
"""

using Pkg
envpath = expanduser("~/envs/dev/")
Pkg.activate(envpath)
using Random
using Distributed
using CUDA


@show Threads.nthreads()   # number of available threads
display(devices())         # list of available GPUs
@show workers()            # list of active workers (only Main process is active)

wpd = 1   # workers per device (gpu) multiplier
addprocs(wpd * length(devices()))
# addprocs(4)
@show workers()            # active workers has changed
@everywhere using CUDA     # make sure CUDA is available on all workers


# assign a device (gpu) to each worker
# c = zip(workers(), devices())
c = zip(workers(), Iterators.cycle(devices()))
asyncmap(c) do (w, d)
    remotecall_wait(w) do
        device!(d)
        @info "Worker $(myid()) uses $d"
    end
end


# setup scripts to be executed
cd(@__DIR__)
@everywhere nepochs = 400
@everywhere debugflag = false
scripts = [
    "unet2.jl",
    "unet4.jl",
    "unet5.jl",
    "mobileunet.jl",
    "espnet.jl",
]
scripts = pwd() * "/" .* scripts
Random.shuffle!(scripts)

@everywhere function executescript(script)
    @info "Worker $(myid()) with device $(device()) is processing $(basename(script))"
    # sleep(rand(1:10))   # simulate some work
    include(script)
    @info "Worker $(myid()) finished $(basename(script))"
end

# start execution of a script as soon as a worker is available
pmap(executescript, scripts)


# cleanup
rmprocs(workers())   # remove workers, remember that workers() is a vector
@show nprocs()       # only main process is left
