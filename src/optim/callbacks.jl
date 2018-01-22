using JLD2, FileIO

"Record standard run metrics to dataframe"
function recordrungen()
  df = DataFrame(runname = Symbol[],
                 iteration = Int[],
                 loss = Float64[],
                 systime = Float64[],
                 input = Vector{Float64}[],
                 output = Vector{Float64}[])
  i = 0
  function recordrun(data)
    push!(df, [i, data.loss, time(), [data.input...], [data.output...]])
    i = i + 1
  end
  df, recordrun
end

"Save dataframe to file"
function savedfgen(name::String, path::String, dfdata::DataFrame)
  function savedf(data)
    jldopen(path, "w") do file
      file[name] = dfdata
    end
    # println("Saving data to $path")
    # @show dfdata
    # save(path, Dict(name => dfdata))
  end
  savedf
end

"Higher order function that makes a callback run just once every n"
function everyn(callback, n::Integer)
  function everyncb(data)
    if data.iter % n == 0
      callback(data)
    end
  end
  return everyncb
end

function printloss(data)
  println("loss: ", data.loss)
end
