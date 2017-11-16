using JLD2
using DataFrames
using AlioAnalysis: getrundata, pathfromgroup, init, optimal, min_domainϵ, min_naive
using Arrows
using AlioZoo

function initdata(group)
  path = pathfromgroup(group)
  @grab data = getrundata(path)
end


"Group runs by `opt[group]`"
function groupruns(rundata, group::Symbol; f=identity)
  grouped = Dict()
  for (opt, df) in rundata
    grp = f(opt[group])
    if grp ∉ keys(grouped)
      grouped[grp] = Dict()
    end
    grouped[grp][opt] = df
  end
  grouped
end

function alldfs(optdfs; filterf=(opt, df)->true)
  alldfs = DataFrame[]
  for (opt, dfs) in optdfs
    @show length(dfs)
    for df in dfs
      if filterf(opt, df)
        push!(alldfs, df)
      end
    end
  end
  alldfs
end

function initoptimal(groupeddata, on::Symbol=:loss)
  toplot = Dict()
  for (fwdarr, optdfs) in groupeddata
    toplot[fwdarr] = []
    for minf in [min_domainϵ, min_naive]
      dfs = alldfs(optdfs; filterf=(opt, df)->opt[:minf] == minf)
      optimalvals = map(df -> optimal(df, on), dfs)
      initvals = map(df -> init(df, on), dfs)
      dataa = Dict(:optimalvals => optimalvals,
                   :initvals => initvals,
                   :minf => minf)
      push!(toplot[fwdarr], dataa)
    end
  end
  toplot
end


function makeplots(toplot)
  hs = []
  for datum in toplot
    @show typeof(datum)
    if !isempty(datum[:optimalvals])
      h = histogram(log(datum[:optimalvals]), label=string("optimal", datum[:minf]), α=0.5)
      push!(hs, h)
    else
      println("Empty Data!:", datum)
    end
    if !isempty(datum[:initvals])
      @show datum[:initvals]
      h = histogram!(log(datum[:initvals]), label=string("init", datum[:minf]), α=0.5)
      push!(hs, h)
    else
      println("Empty Data!", datum)
    end
  end
  hs
end

function analysis(group)
  # Get the data
  data = initdata(group)

  # Group by the forward arrow
  groupeddata = groupruns(data, :fwdarr, f=name)

  initoptimal(groupeddata)
end
