using DataFrames
using Plots
using Spec
RunData = Dict{Symbol, Any}

# What should be that data format passed from combine?
# Probably a dataframe

"Was dataset `df` created from run `rd`"
function isdatafromrun(df::DataFrame, rdrunname::Symbol)
  all((runname == rdrunname for runname in df[:runname]))
end

"Was dataset `df` created from run `rd`"
isdatafromrun(df::DataFrame, rd::RunData) = isdatafromrun(df, rd[:runname])

"Rename each column `df` (if column name ∈ `torename`) to `colname_runname`"
function renamecolsbyrun(df::DataFrame, torename::Vector{Symbol}, runname::Symbol)
  @pre all(map(isdatafromrun, dfs, runname))
  f(colname) = colname ∈ torename ? Symbol(colname, :_, runname) : colname
  rename(f, df)
end

"""
Join and rename multiple `DataFrame`s `dfs` from rundata

- join on `on`
"""
function combinedata(dfs::Vector{DataFrame},  # FIXME: Rename more meaningful
                     rds::Vector{RunData},
                     on::Symbol,
                     select::Vector{Symbol})
  @pre all(map(isdatafromrun, dfs, rds))
  dfs = map(dfs, [rd[:runname] for rd in rds]) do df, runname
    renamecolsbyrun(df, select, runname)
  end
  # TODO: Generalize to n dfs
  join(dfs[1], dfs[2], on = on, kind = :outer)
end

"LOL"
function plotlinechart(df::DataFrame, )
  plot(df[:iteration], [data[:iteration], data[:iteration]])
end

function summarize(df::DataFrame)
  plot(iterations, losses)
end

function test()

  # Run 1
  n = 1000
  rundata1 = RunData(:runname => :exrun1,
                     :time => now(),
                     :desc => "An example run",
                     :completed => false)

  df1 = DataFrame(runname = [:exrun1 for i=1:n],
                  iteration = 1:n,
                  loss = [exp(-0.001i)*rand() for i=1:n],
                  time = 10*[exp(-0.001i)*rand() for i=1:n])

  # Run 2
  n = 700
  rundata2 = RunData(:runname => :exrun2,
                     :time => now(),
                     :desc => "An example run")

  df2 = DataFrame(runname = [:exrun2 for i=1:n],
                  iteration = 1:n,
                  loss = [exp(-0.005i)*rand() for i=1:n],
                  time = 10*[exp(-0.005i)*rand() for i=1:n],
                  error = 10*[exp(-0.005i)*rand() for i=1:n])

  # TODO: Need another stage to manage all the data frames
  rds = [rundata1, rundata2]
  dfs = [df1, df2]

  data = combinedata(dfs, rds, :iteration, [:loss, :time])
  # plotlinechart(data)
  # summarize(data)
end
