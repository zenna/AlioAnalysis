# Actual concrete analyses
# Requires fields for RunData
# Helper filter data frames based on rundata conditions
# Encaspulate runs into some kind of structure for consistency
# Better combinators for Arrows


RunData = Dict{Symbol, Any}

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
