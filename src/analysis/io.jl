function parsedata(file; optext = "opt", jldext = "jld2", datakey = "rundata")
  if extension(file) == optext
    return AlioAnalysis.loadopt(file)
  elseif extension(file) == jldext
    local rundata
    jldopen(file, "r") do file
      rundata = file[datakey]
    end
    return rundata
  else
    @show extension(file)
    @assert false file
  end
end

"Get all rundata files within `dir`"
function getrundata(dir)
  optrundata = Dict()
  for (root, dirs, files) in walkdir(dir)
    for dir in dirs
      println(joinpath(root, dir)) # path to directories
    end
    if all(ext -> any(file -> extension(file) == ext, files), ["opt", "jld2"])
      opt = parsedata(joinpath(root, "options.opt"))
      optrundata[opt] = []
      for file in files
        if extension(file) == "jld2"
          data = parsedata(joinpath(root, file))
          push!(optrundata[opt], data)
        end
      end
    end
  end
  optrundata
end

"Assumes only one dot"
extension(file) = basename(file)[search(basename(file), ".")[1]+1:end]
