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
    push!(df, [i, data.loss, time(), [data.input...], [data.output...]])
    i = i + 1
  end
  df, stdupdate
end

function min_naive(carr::Arrow, inputs...)
  @show naive_loss, ϵid = Arrows.naive_loss(carr, inputs...)
  ϵprt = ◂(naive_loss, ϵid)
  df, update_data = stdrecordcb()
  Arrows.optimize(naive_loss, ▸(naive_loss), ϵprt, rand(2); callbacks = [update_data])
  [df]
end

"Minimize `arr` using to"
function min_domainϵ(arr::Arrow, outs...)
  metadata = @NT(runname="runname")
  dmloss, ϵid = domain_ovrl(arr)
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
function scattermany(points, results, newplot::Bool, markershape=:circle)
  @show size(points)
  @show size.(results)
  lb = 1
  local plot
  for res in results
    scat = lb == 1 &&newplot ? scatter : scatter!
    ub = lb+size(res, 1)-1
    @show lb, ub
    pts = points[lb:ub, :]
    plot = scat(pts[:,1], pts[:,2], markershape=markershape)
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
  x = rand(length(▸(carr, !is(θp))))
  y = arr(y)
  naivedata = (dfs->manyjoin(:iteration, dfs...)).([min_naive(carr) for i = 1:nruns])
  # domaindata = min_domainϵ(carr)
  domaindata = (dfs->manyjoin(:iteration, dfs...)).([min_domainϵ(carr) for i = 1:nruns])

  concatted1 = vcat([df[:loss] for df in naivedata]...)
  concatted2 = vcat([df[:loss] for df in domaindata]...)
  concatted = vcat(concatted1, concatted2)
  points = tsne(Array(concatted), (x, y)->(sum(abs.(x-y))))
  @show size(concatted)
  @show size(concatted1)
  @show size(concatted2)
  scattermany(points[1:length(concatted1), :], naivedata, true, :circle)
  scattermany(points[length(concatted1):end, :], domaindata, false, :xcross)
end

function manyjoin(on::Symbol, dfs::DataFrame...)
  x = first(dfs)
  for df in dfs[2:end]
    @show typeof(x)
    @show typeof(df)
    x = join(x, df, on=on)
  end
  x
end
