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
       train
       plus,
       optimizerun,
       genloss

include("util.jl")          # Genral Utils
include("callbacks.jl")     # Functions passed to optimizxation
include("opts.jl")          # Standard options and option samplers
include("search.jl")        # Methods

# Runs
include("plots/space.jl")
include("plots/warp.jl")
include("query.jl")
include("run.jl")
end
