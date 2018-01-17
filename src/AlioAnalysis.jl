"""
Alio Analysis is divided into three kinds of things
- General tools for data analyis and plotting
- Tools for running optimization (while collecting data) using both
  - function optimization with tensor flow
  - pointwise optimization with nlopt
- `runs` which test particular popreties of Arrows often from AlioZoo
"""
module AlioAnalysis

using Arrows
using DataFrames
using Plots
using NamedTuples
using Query
using DataArrays
using TSne
using NLopt
# using JLD2

export saveopt,
       log_dir,
       prodsample,
       makeloss,
       savedata,
       genorrun,
       dorun,
       netpi,
       invnet,
       train,
       plus,
       optimizerun,
       genloss,
       Options

include("util/misc.jl")             # Genral Utils
include("analysis/space.jl")
include("analysis/io.jl")

include("transforms.jl")

include("optim/callbacks.jl")
include("optim/functionwise.jl")
include("optim/pointwise.jl")
include("optim/loss.jl")
end
