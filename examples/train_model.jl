@info "Project start"
cd(@__DIR__)

### code arguments
cudadevice = 0
nepochs    = 10
debugflag  = true

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"


### libs
using Pkg
envpath = expanduser("~/envs/dev/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)

using Flux
import Flux: leakyrelu
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Statistics: mean

# private libs
using TinyMachines; const tm=TinyMachines
using PascalVocTools; const pv=PascalVocTools   # https://github.com/cirobr/PascalVocTools.jl
@info "environment OK"


### constants
const KB = 1024
const MB = KB * KB
const GB = KB * MB


### folders
outputfolder = script_name[1:end-3] * "/"

# pwd(), homedir()
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "~/projects/knowledge-distillation/dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist
@info "folders OK"


### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-coi-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.segmented .== 1,:]

fpfn = expanduser(datasetpath) * "dfvalid-coi-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
dfvalid = dfvalid[dfvalid.segmented .== 1,:]
@info "datasets OK"


########### debug ############
if debugflag
      dftrain = first(dftrain, 5)
      dfvalid = first(dfvalid, 2)
      minibatchsize = 1
      epochs  = 2
else
      minibatchsize = 4
      epochs  = nepochs
end
##############################


# create tensors
@info "creating tensors..."
Xtr = Images.load(expanduser(dftrain.X[1]))
dims = size(Xtr)
Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)
Xtrain = Array{Float32, 4}(undef, (dims...,3,Ntrain))
ytrain = Array{Bool, 4}(undef, (dims...,C,Ntrain))
Xvalid = Array{Float32, 4}(undef, (dims...,3,Nvalid))
yvalid = Array{Bool, 4}(undef, (dims...,C,Nvalid))

dfs = [dftrain, dfvalid]
Xouts = [Xtrain, Xvalid]
youts = [ytrain, yvalid]

for (df, Xout, yout) in zip(dfs, Xouts, youts)   # no @floop here
      N = size(df, 1)
      @floop for i in 1:N
            local fpfn = expanduser(df.X[i])
            img = Images.load(fpfn)
            img = img |> channelview |> x -> permutedims(x, (2,3,1)) .|> Float32
            Xout[:,:,:,i] = img

            local fpfn = expanduser(df.y[i])
            mask = Images.load(fpfn)
            mask = pv.voc_rgb2classes(mask)
            mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            mask = permutedims(mask, (2,3,1)) .|> Bool
            yout[:,:,:,i] = mask
      end
end
@info "tensors OK"


# dataloaders
Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                              batchsize=minibatchsize,
                              shuffle=true) |> gpu
validset = Flux.DataLoader((Xvalid, yvalid)) |> gpu
@info "dataloader OK"


# check memory requirements
Xtr = Xtrain[:,:,:,1:1] |> gpu
ytr = ytrain[:,:,:,1:1] |> gpu

dpsize = sizeof(Xtr) + sizeof(ytr)
dpGB = dpsize / GB
@info "datapoints in trainset = $(size(Xtrain,4))"
@info "datapoints in 1 GB = $(1 / dpGB)"


### model
Random.seed!(1234)   # to enforce reproducibility
modelcpu = UNet2(3,C; activation=leakyrelu, alpha=1, verbose=false)
# modelcpu = UNet4(3,C; activation=leakyrelu, alpha=1, verbose=false)
# modelcpu = UNet5(3,C; activation=leakyrelu, alpha=1, verbose=false)
# modelcpu = MobileUNet(3,C; verbose=false)
# modelcpu = ESPnet(3,2; activation=leakyrelu, alpha2=3, alpha3=4, verbose=false)

model = modelcpu |> gpu
@info "model OK"


### check for matching between model and data
@assert size(model(Xtr)) == size(ytr) || error("model/data features do not match")
@info "model/data matching OK"
###


# loss function
loss(m, X, y) = Flux.crossentropy(m(X), y, dims=3)   # m is the first argument
@info "loss function OK"


# optimizer
opt = Flux.Adam()
opt_state = Flux.setup(opt, model)
@info "optimizer OK"


### training
@info "start training ..."
Random.seed!(1234)   # to enforce reproducibility
validloss = Vector{Float32}(undef, length(validset))

for i in 1:epochs
      Flux.train!(loss, model, trainset, opt_state)

      for (i, (X,y)) in enumerate(validset)
            validloss[i] = loss(model, X, y)
      end
      @info "epoch $i, loss = $(mean(validloss))"
end
@info "project finished"
