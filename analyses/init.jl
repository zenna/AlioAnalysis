using JLD2
using DataFrames
using AlioAnalysis: getrundata, pathfromgroup, init, optimal
using Arrows
using AlioZoo

"Get data for initialization analysis"
function initdata(group)
  path = pathfromgroup(group)
  data = getrundata(path)
end

function spectogramdata(group)
  path = pathfromgroup(group)
  data = AlioAnalysis.getrundata2(path)
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
    for df in dfs
      if filterf(opt, df)
        push!(alldfs, df)
      end
    end
  end
  alldfs
end

# ba = 2
function initoptimal(fgroupeddata, on::Symbol=:loss)
  # 3 + x = 4
  toplot = Dict()
  for (fwdarr, optdfs) in groupeddata
    toplot[fwdarr] = Dict()
    for minf in [min_domainϵ, min_naive]
      dfs = alldfs(optdfs; filterf=(opt, df)->opt[:minf] == minf)
      optimalvals = map(df -> optimal(df, on), dfs)
      initvals = map(df -> init(df, on), dfs)
      dataa = Dict(:optimalvals => optimalvals,
                   :initvals => initvals,
                   :minf => minf)
      toplot[fwdarr][minf] = dataa
    end
  end
  toplot
end

function makeplots(toplot)
  labels = RowVector([])
  logvals = []
  for (minf, datum) in toplot
    labels = hcat(labels, string("optimal", minf))
    push!(logvals, log(datum[:optimalvals]))
    labels = hcat(labels, string("init", minf))
    push!(logvals, log(datum[:initvals]))
  end
  histogram(logvals, labels=labels, α=0.5, line=(0, :dash))
  xaxis!("Loss")
  title!("Histogram of loss values")
end

function summarizeloss(groupeddata)
  df = DataFrame(Fwdarr = [],
                 Test = [],
                 Optimmean = Float64[],
                 Optimvar = Float64[],
                 Initmean = Float64[],
                 Initvar = Float64[])
  for (fwdarr, testtype) in groupeddata
    for (datakind, x) in testtype
      push!(df,
            [fwdarr,
             datakind,
             mean(x[:optimalvals]),
             var(x[:optimalvals]),
             mean(x[:initvals]),
             var(x[:initvals])])
      for (k,v) in x
      end
    end
  end
  df
end

"Get mean and variance values"
function meansvars(df)
  by(df, :Test) do df
     DataFrame(optmean = mean(df[:Optimmean]),
               optvar = var(df[:Optimvar]),
               initmean = mean(df[:Initmean]),
               initvar = var(df[:Initvar]))
  end
end

function analysis(group)
  # Get the data
  data = initdata(group)
  # Group by the forward arrow
  groupeddata = groupruns(data, :fwdarr, f=name)
  toplot = initoptimal(groupeddata)
  # make a plot
  makeplots(first(values(toplot)))
end
