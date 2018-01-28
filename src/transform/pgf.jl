function okok(tabv::Arrows.TraceAbValues, arr::Arrow)::NmAbValues
  # Assume theres only one tabv that corresponds to 
  tsprts_set = map(Arrows.trace_sub_ports, keys(tabv))
  ids = Int[]
  for sprt in ⬨(arr)
    idx = findfirst(tsprts_set) do tsprts
      sprt ∈ map(Arrows.sub_port, tsprts)
    end
    @assert idx != 0
    push!(ids, idx)
  end
  abv = collect(values(tabv))
  @show ids
  Arrows.NmAbValues(port_sym_name(prt) => abv[ids[prt.port_id]] for prt in ⬧(arr))
end

"Computes `δ(pgf(x), n(f(x))`"
function δpgfx_ny_arr(f::Arrow, xabv::XAbValues)
  invf = invert(f, inv, xabv)
  tabv = traceprop!(invf, xabv)
  nabv = okok(tabv, invf)
  n = pslnet(invf)
  lossarr = δny_x_arr(n; nm = :δpgfx_ny_arr)
  lossarr, n, nabv
end

"Iterator of `[f(x), pgf(x)]`  values from `xgenss`: Iterator over inputs "
function x_to_y_θ_gen(pgfarr::Arrow, xgen)
  θprts, yprts = Arrows.partition(is(θp), ◂(pgfarr))
  @grab pgfarr
  @grab θoids = out_port_id.(θprts)
  @grab yoids = out_port_id.(yprts)
  @grab pgfjl = il(Arrows.splat(julia(pgfarr)))
  function splitoutvals(xs::Tuple)
    [[xs[i] for i in yoids]; [xs[i] for i in θoids]]
  end
  imap(splitoutvals ∘ pgfjl, xgen)
end

"""
Train preimage attack network using pgf

Given forward arrow `f`
- Generate input data `x`
- Generate output data `y = f(x)`
- Learn mapping `net: y -> x` that minimizes argmin(f(net(y)))
"""
function trainpgfnet(lossarr::Arrow,
                     n::Arrow,
                     y_θ_gen,
                     xabv,
                     optimtarget,
                     template;
                     optimizeargs...)
  # @pre same([n◂(lossarr), n▸(n)]) # What should these be?
  nnettarr = first(Arrows.findtarrs(lossarr, n))
  @grab nnettarr
  @grab xabv
  @grab tabv = Arrows.tabvfromxabv(nnettarr, xabv)

  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             optimtarget,
             template,
             ingens = y_θ_gen,
             xabv = tabv;
             optimizeargs...)
end


using TensorFlowTarget

"Cutie pie"
function test_pgf_training(f = TestArrows.xy_plus_x_arr())
  batch_size = 32
  sz = [batch_size, 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(f))
  pgff = pgf(f, Arrows.pgf, xabv)
  lossarr, n, xabv = δpgfx_ny_arr(f, xabv)
  @grab lossarr
  @grab n
  @grab xabv
  
  # @grab tabv
  # @assert false
  xgen = Sampler(()->[rand(sz...) for i = 1:n▸(f)])
  y_θ_gen = x_to_y_θ_gen(pgff, xgen)
  @grab xgen
  @grab y_θ_gen
  trainpgfnet(lossarr,
              n,
              y_θ_gen,
              xabv,
              TensorFlowTarget.TFTarget,
              TensorFlowTarget.mlp_template)
end
