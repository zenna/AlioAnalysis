"Standard callback that records"
function stdrecordcb()
  df = DataFrame(iteration = Int[],
                 loss = Float64[],
                 systime = Float64[],
                 input = Vector{Float64}[],
                 output = Vector{Float64}[])
  i = 0
  function stdupdate(data)
    push!(df, [i, data.loss, time(), [data.input...], [data.output...]])
    i = i + 1
  end
  df, stdupdate
end

"Save loss data"
function savedata()
  df = DataFrame(iteration = Int[],
                 loss = Float64[],
                 systime = Float64[])
  function stdupdate(data)
    push!(df, [data.i, data.loss, time()])
  end
  df, stdupdate
end
