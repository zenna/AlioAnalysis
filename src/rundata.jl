RunData = Dict{Symbol, Any}

"Return list of pairs `(df::DataFrame, rd::RunData)` where `rundata(df) == dfs`"
function zipdfrds(dfs::Vector{DataFrame}, rds::Vector{RunData})
  dfrds = map(df->rundata(df, rds), dfs)
  zip(dfs, dfrds)
end

"Was dataset `df` created from run `rd`"
function isdatafromrun(df::DataFrame, rdrunname::Symbol)
  @show names(df)
  all((runname == rdrunname for runname in df[:runname]))
end

"Was dataset `df` created from run `rd`"
isdatafromrun(df::DataFrame, rd::RunData) = isdatafromrun(df, rd[:runname])

"Rename each column `df` (if column name ∈ `torename`) to `colname_runname`"
function renamecolsbyrun(df::DataFrame, torename::Vector{Symbol}, runname::Symbol)
  @pre isdatafromrun(df, runname)
  f(colname) = colname ∈ torename ? Symbol(colname, :_, runname) : colname
  rename(f, df)
end

"""
Join and rename multiple `dfs::DataFrame` from run data
- join on `on`
"""
function combinedata(dfs::Vector{DataFrame},  # FIXME: Rename more meaningful
                     rds::Vector{RunData},
                     on::Symbol,
                     select::Vector{Symbol})
  @pre all(map(isdatafromrun, dfs, rds))
  @pre all((all(haskey(df, colnm) for colnm in select) for df in dfs))

  # Rename columns so e,g, loss -> run_1234_loss foreach df
  dfs = map(dfs, [rd[:runname] for rd in rds]) do df, runname
    @show names(df), "a", on
    df = df[[:runname; on; select]]
    @show names(df), "b", on
    renamecolsbyrun(df, select, runname)
  end
  # TODO: Generalize to n dfs
  join(dfs[1], dfs[2], on = on, kind = :outer)
  # [:loss_ada, :stats_time]
end

function plotlinechart(df::DataFrame, xnm::Symbol, ynms::Vector{Symbol})
  aba = [df[ynm] for ynm in ynms]
  @grab aba
  plot(df[xnm], [df[ynm] for ynm in ynms])
end
