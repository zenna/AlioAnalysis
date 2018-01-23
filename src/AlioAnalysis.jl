"""
AlioAnalysis has four pillars

1. Arrows
  An `Arrow` is a dataflow program from `Arrows.jl`

2. Runs
  Arrows are executed in a run.

3. Datasets
  Runs accumulate data which is stored in datasets

4. Reports
  Datasets are analyzed, and summarized into graphs, tables, statistics.
  Reports are the desired output of using AlioAnalysis

The following are typically many-many relationships:
  (Arrow, Runs)
  (Runs, Datasets)
  (Datasets, Reports)

For example there can be many arrows involved in one run, and each arrow may be
inboled in many runs.
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
using Spec
# using JLD2

export savedict,
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
       Options,
       savedf,
       loaddf,
       loadrundata,
       walkrundata,
       walkdfdata,
       rundata,
       RunData,
       combinedata,
       plotlinechart

include("util/misc.jl")             # Genral Utils
include("rundata.jl")
include("analysis/space.jl")
include("analysis/io.jl")

include("transforms.jl")

include("optim/callbacks.jl")
include("optim/functionwise.jl")
include("optim/pointwise.jl")
include("optim/loss.jl")
end
