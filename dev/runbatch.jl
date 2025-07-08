"""
Author: cirobr@GitHub
Date: 07-03-2025

Template for executing multiple scripts in series.
    * Each project lives on a separate folder.
"""

### arguments
envpath    = "./"
cudadevice = 1
nepochs    = 1 #400
debugflag  = true

### environment
using Pkg
envpath = expanduser(envpath)
Pkg.activate(envpath)

# folders
scripts = [
    "espnet.jl",
]
folders = [script[1:end-3] for script in scripts]
models = @. "models/" * folders * "/"
tblogs = @. "tblogs/" * folders * "/"

@info "Project batch started"
@. rm(models, force=true, recursive=true)
@. rm(tblogs, force=true, recursive=true)
@. include(scripts)
@info "Project batch completed!"
