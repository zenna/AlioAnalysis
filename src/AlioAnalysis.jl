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

include("util.jl")          # Genral Utils
include("callbacks.jl")     # Functions passed to optimizxation
include("opts.jl")          # Standard options and option samplers
include("search.jl")        # Methods

# Runs
ba = 3
include("runs/init.jl")

include("plots/space.jl")
include("plots/warp.jl")
include("query.jl")
end
