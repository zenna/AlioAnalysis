module AlioAnalysis

using Arrows
using DataFrames
using Plots
using NamedTuples
using Query
using DataArrays
using TSne
using NLopt

include("plots/space.jl")
include("plots/warp.jl")
include("callbacks.jl")
include("query.jl")
end
