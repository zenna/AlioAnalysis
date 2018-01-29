# DataFrames related utilities


"Recursive join on `on` of many dataframes `dfs`"
function manyjoin(on::Symbol, dfs::DataFrame...)
  x = first(dfs)
  for df in dfs[2:end]
    x = join(x, df, on=on)
  end
  x
end

"Append one or more dataframes to `df1`"
function manyappend(df1::DataFrame, df2::DataFrame, dfs::DataFrame...)
  df1 = deepcopy(df1)
  dfs = vcat(df2, [dfs...])
  foreach(df -> append!(df1, df), dfs)
  df1
end
