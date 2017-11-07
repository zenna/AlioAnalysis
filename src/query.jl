using Query
using DataFrames
using DataArrays
using TSne
using Plots

df = DataFrame(name=["John", "Sally", "Kirk"], age=[23., 42., 59.], children=[3,5,2])
losses = DataFrame(run=["run1", "run2", "run3"], )

function join(df)
  x = @from i in df begin
      @where i.age > 0
      @select {i.name, i.children}
      @collect DataFrame
  end
end

# 1. Design data layout
# 2. Fix Tsne
# 3. # negro please

function test_analysis(carr=xy_plus_x_arr())
  # 1 invert it
  inv = aprx_invert(carr)
end

using NLopt

function appenddata()
  df = DataFrame(iteration = Int[],
                 loss = Float64[],
                 systime = Float64[],
                 input = Vector{Float64}[],
                 output = Vector{Float64}[])
  i = 0
  function update_data(data)
    push!(df, [i, data.loss, time(), [data.input...], [data.output...]])
    i = i + 1
  end
  df, update_data
end

function test_tsne(carr = TestArrows.xy_plus_x_arr())
  @show naive_loss, ϵid = Arrows.naive_loss(carr, 1.0)
  ϵprt = ◂(naive_loss, ϵid)
  df, update_data = appenddata()
  Arrows.optimize(naive_loss, ▸(naive_loss), ϵprt, rand(2); callbacks = [update_data])
  df
end

function tsne_plot(carr::Arrow, dist::Symbol, nruns::Integer)
  results = [test_tsne(carr) for i = 1:nruns]
  concatted = vcat([df[dist] for df in results]...)
  points = tsne(Array(concatted), (x, y)->(sum(abs.(x-y))))
  @show size(points)
  @show size.(results)
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
