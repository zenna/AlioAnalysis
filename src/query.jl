using Query
using DataFrames
using DataArrays
using TSne
using Plots
using Arrows
using NLopt

# function join(df)
#   x = @from i in df begin
#       @where i.age > 0
#       @select {i.name, i.children}
#       @collect DataFrame
#   end
# end

"Standard callback that records"
function stdrecordcb()
  df = DataFrame(iteration = Int[],
                 loss = Float64[],
                 systime = Float64[],
                 input = Vector{Float64}[],
                 output = Vector{Float64}[])
  i = 0
  function stdupdate(data)
   println("AGOT HEREB")
    push!(df, [i, data.loss, time(), [data.input...], [data.output...]])
    i = i + 1
  end
  df, stdupdate
end

function min_naive(carr = TestArrows.xy_plus_x_arr())
  @show naive_loss, ϵid = Arrows.naive_loss(carr, 1.0)
  ϵprt = ◂(naive_loss, ϵid)
  df, update_data = stdrecordcb()
  Arrows.optimize(naive_loss, ▸(naive_loss), ϵprt, rand(2); callbacks = [update_data])
  [df]
end

"Minimize `arr` using to"
function min_domainϵ(arr = TestArrows.xy_plus_x_arr(), inputs=rand(length(▸(arr, !is(θp)))))
  outs = arr(inputs...)
  @show dmloss, ϵid = domain_ovrl(arr)
  invarr = invert(arr)
  aprx_totalize!(invarr)
  idloss = id_loss(arr, invarr)

  # Callback generator which records domain and id loss
  function domaincallback()
    df2 = DataFrame(iteration = Int[],
                   idloss = Float64[],
                   domainloss = Float64[])
    i = 0
    function domainupdate(data)
      println("GOT HEREB")
      # compute the domain loss and id loss
      dm_error = dmloss(data.input...)[ϵid]
      id_error = idloss(data.input...)[1]
      push!(df2, [i, dm_error, id_error])
      i = i + 1
    end
    df2, domainupdate
  end

  
  df, stdupdate = stdrecordcb()
  # @show update_data(3)
  domaindf, domupdate = domaincallback()
  # @show update_data(3)

  init = [outs..., rand(length(▸(dmloss, is(θp))))...]
  optimize(dmloss, ▸(dmloss, is(θp)), ◂(dmloss, ϵid), init;
            callbacks=[stdupdate, domupdate])
  [df, domaindf]
end

"Scatter many points"
function scattermany(points)
  lb = 1
  local plot
  for res in results
    scat = lb == 1 ? scatter : scatter!
    ub = lb+size(res, 1)-1
    @show lb, ub
    pts = points[lb:ub, :]
    plot = scat(pts[:,1], pts[:,2])
    lb = ub + 1
  end
  plot
end

# function ok(carr::Arrow, dist::Symbol, nruns::Integer, gendata::Function)
#   results = [gendata(carr) for i = 1:nruns]
#   concatted = vcat([df[dist] for df in results]...)
#   points = tsne(Array(concatted), (x, y)->(sum(abs.(x-y))))
#   @show size(points)
#   @show size.(results)
# end

function compare_pi_naive(carr::Arrow, nruns::Integer)
  naivedata = min_naive(carr)
  domaindata = min_domainϵ(carr)
  naivedata, domaindata
end

function manyjoin(on::Symbol, dfs::DataFrame...)
  x = first(dfs)
  for df in dfs[2:end]
    @show typeof(x)
    @show typeof(df)
    x = join(x, df, on)
  end
  x
end
