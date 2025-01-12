@info "Project start"
cd(@__DIR__)

### libs
using Pkg
envpath = expanduser("~/envs/dev/")
Pkg.activate(envpath)

using CUDA

using Flux
import Flux: relu, leakyrelu, softmax
using Images
using DataFrames
using CSV
using JLD2
using BenchmarkTools

# private libs
using TinyMachines; const tm=TinyMachines
using PreprocessingImages; const p=PreprocessingImages
using PascalVocTools; const pv=PascalVocTools
using LibFluxML
# using LibCUDA
@info "environment OK"

### folders
# pwd(), homedir()
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "../dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-coi-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.segmented .== 1,:]

k = 1
Xtrain = dftrain.X[k]
ytrain = dftrain.y[k]

fpfn = expanduser(Xtrain)
Xtr = Images.load(fpfn)
Xtr = p.color2Float32(Xtr)
Xtr = reshape(Xtr, (size(Xtr)..., 1))

fpfn = expanduser(ytrain)
ytr = Images.load(fpfn)
ytr = pv.voc_rgb2classes(ytr)
ytr = Flux.onehotbatch(ytr, [0,classnumbers...],0)
ytr = permutedims(ytr, (2,3,1))
@info "tensors OK"

### model
modelcpu = tm.UNet5(3,C; activation=leakyrelu, alpha=1, verbose=false)
# fpfn = expanduser("")
# LibML.loadModelState!(fpfn, modelcpu)
model    = modelcpu |> gpu
@info "model OK"

# setup benchmarking
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 180

### benchmarking
@info "benchmarking forward pass ..."
Xtr = Xtr |> gpu

bm = @benchmark model($Xtr)

fn = basename(@__FILE__)
fn = replace(fn, ".jl" => ".jld2")
JLD2.save_object(fn, bm)
# bm2 = JLD2.load_object(fn)
@info "benchmarking OK"

# results
println()
display(bm)
# dump(bm)

println()
println("minimum time:")
display(minimum(bm))

println()
println("median time:")
display(median(bm))
