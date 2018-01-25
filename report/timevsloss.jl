using Plots
plotlyjs()
using Arrows
using AlioAnalysis
using Spec
import AlioAnalysis: zipdfrd, walkrundata, walkdfdata
using DataFrames


function plotlinechart(df::DataFrame, xnm::Symbol, ynms::Vector{Symbol})
  aba = [df[ynm] for ynm in ynms]
  @grab aba
  plot(df[xnm], [df[ynm] for ynm in ynms])
end

function plot_iteration_loss(dfs::Vector{DataFrame}, rds::Vector{RunData})
  @pre all((haskey(df, :loss) for df in  dfs))
#   @pre map(isdatafromrun
  its = []
  losses = []
  for df in dfs
    push!(its, df[:iteration])
    push!(losses, df[:loss])
  end
  plot(its, losses, xlabel="Iteration", ylabel="Loss")
end

function data(path)
  rds = walkrundata(path)
  dfs = walkdfdata(path)
  zipped = zipdfrd(dfs, rds)
  plot_iteration_loss(zipped.a, zipped.b)
end