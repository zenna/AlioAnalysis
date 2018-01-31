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
using NamedTuples
using Query
using NLopt
using Spec
using ProgressMeter
import IterTools: imap

import Arrows: Err, add!, idϵ, domϵ, TraceSubArrow, trace_port, TraceValue, TraceAbVals, pfx, reduce_sum, reduce_mean
# using JLD2

import Base: gradient

export savedict,
       log_dir,
       prodsample,
       genorrun,
       dorun,
       netpi,
       invnet,
       dispatchruns,
       plus,
       optimizenet,
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
       plotlinechart,

       # Optimization
       optimize,
       verify_loss,
       verify_optim,
       recordrungen,
       everyn,
       savedfgen,
       printloss,
       linearstring,
       port_sym_names,
       pianet,
       fxgen,
       trainpianet,
       Sampler

include("util/misc.jl")             # Genral Utils
include("util/dataframes.jl")       # DataFrames utils
include("util/generators.jl")       # Iterators / Generators 
include("util/dispatch.jl")         # Running jobs

include("rundata.jl")
include("analysis/space.jl")
include("analysis/io.jl")

include("transform/common.jl")
include("transform/supervised.jl")
include("transform/pia.jl")
include("transform/reparam.jl")
include("transform/pgf.jl")

include("optim/callbacks.jl")
include("optim/optimizenet.jl")
include("optim/gradient.jl")
include("optim/loss.jl")
include("optim/optimize.jl")
include("optim/pointwise.jl")
include("optim/util.jl")

end
