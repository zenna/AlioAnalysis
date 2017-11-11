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


function get_data(arr::Arrow, nruns::Integer, doruns::Function...)
  dfs = DataFrame[]
  for dorun in doruns
    # do nruns runs and
    rundata = Vector{DataFrame}[dorun(arr) for i = 1:nruns]
    rundata = map(joincallbacks, rundata)
    joinruns(rundata)
    push!(dfs, joineddfs)
  end
  dfs
end

function compare(arr::Arrow, nruns::Integer, funcs::Function...)
  dfs = get_data(arr, nruns, funcs)
  alllosses = Float64[]
  for df in dfs
    losses = vcat(df[:loss] for df in joineddf)
  end

  points = tsne(Array(alllosses), (x, y)->(sum(abs.(x-y))))
end

"Join data from different callbacks from a run on :iteration"
joincallbacks(dfs::DataFrame...) = manyjoin(:iteration, dfs...)

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

"Recursive join on `on` of many dataframes `dfs`"
function manyjoin(on::Symbol, dfs::DataFrame...)
  x = first(dfs)
  for df in dfs[2:end]
    x = join(x, df, on=on)
  end
  x
end

"Append one or more dataframes to `df1`"
function manyappend!(df1::DataFrame, df2::DataFrame, dfs::DataFrame...)
  dfs = vcat(df2, [dfs...])
  foreach(df -> append!(df1, df), dfs)
  df1
end
