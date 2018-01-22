"Extension of `file`"
function extension(file)::String
  # HACK: This is not robust
  if !contains(file, ".")
    ""
  else
    basename(file)[search(basename(file), ".")[1]+1:end] #
  end
end

isrundatafile(file) = extension(file) == "rd"
isdffile(file) = extension(file) == "jld2"

"Save `df` as jld2 to `path`"
function savedf(path::String, df::DataFrame)
  @pre isdffile(path)
  jldopen(path, "w") do file
    file["dataframe"] = df
  end
end

"Load a single `DataFrame` to a JLD2 file at location `path`"
function loaddf(path::String)::DataFrame
  @pre ispath(path)
  local rundata = Dict()
  jldopen(path, "r") do file
    for k in keys(file)
      rundata[k] = file[k]
    end
  end
  return rundata["dataframe"]
end

"Save `Dict` to file"
function savedict(path, opt)
  open(path, "w") do f
    serialize(f, opt)
  end
end

"Load `Dict` from a file"
function loaddict(path)
  deserialize(open(path))
end

"Parse a rundata file"
function loadrundata(path::String)::RunData
  @pre isrundatafile(path)
  loaddict(path)
end


"""
Search through path and find `DataFrame`s and `RunData`s satisfying `filefilter`
"""
function walkload(searchpath::String, isgood, loaddata)
  data = []
  for (root, dirs, files) in walkdir(searchpath)
    for file in files
      extension(file)
      if isgood(file)
        push!(data, loaddata(joinpath(root, file)))
      end
    end
  end
  data
end

"Loads all rundata files in `searchpath`"
walkrundata(searchpath)::Vector{RunData} = walkload(searchpath, isrundatafile, loadrundata)

"Loads all dataframes in `searchpath`"
walkdfdata(searchpath)::Vector{DataFrame} = walkload(searchpath, isdffile, loaddf)

"Is `xs` a singleton (element of one value)?"
issingleton(xs::Set) = length(xs) == 1

"Is set corresponding `xs` (i.e. ignores duplicates) a singleton?"
issingleton(xs::Vector) = length(unique(xs)) == 1

"Runname of rundata that created `df`"
function runname(df::DataFrame)::Symbol
  @pre issingleton(df[:runname])
  df[:runname][1]
end

"Rundata of `df` from set of `rds::RunData`"
function rundata(df::DataFrame, rds::Vector{RunData})::RunData
  @pre issingleton(filter(rd -> rd[:runname] == runname(df), rds))
  first(filter(rd -> rd[:runname] == runname(df), rds))
end
