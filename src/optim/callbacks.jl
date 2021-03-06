using JLD2, FileIO

"Record standard run metrics to dataframe"
function recordrungen(runname::Symbol)
  df = DataFrame(runname = Symbol[],
                 iteration = Int[],
                 loss = Float64[],
                 systime = Float64[],
                 input = Vector{Float64}[],
                 output = Vector{Float64}[])
  i = 0
  function recordrun(cbdata)
    row = [runname, i, cbdata.loss, time(), [cbdata.input...], [cbdata.output...]]
    push!(df, row)
    i = i + 1
  end
  df, recordrun
end

"Save dataframe to file"
function savedfgen(path::String, df::DataFrame)
  cbdata -> savedf(path, df)
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
