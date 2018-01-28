"Computes `δ(pgf(x), n(f(x))`"
function δpgfx_ny_arr(f::Arrow, xabv::XAbValues)
  invf = invert(f, inv, xabv)
  xabv = traceprop!(invf, xabv)
  n = pslnet(invf)
  lossarr = δny_x_arr(n; nm = :δpgfx_ny_arr)
  lossarr, n, xabv
end

"Iterator of `[f(x), pgf(x)]`  values from `xgenss`: Iterator over inputs "
function x_to_y_θ_gen(pgfarr::Arrow, xgens)
  θprts, yprts = Arrows.partition(is(θp), ◂(pgfarr))
  θoids = out_port_id.(θprts)
  yoids = out_port_id.(yprts)
  pgfjl = il(julia(pgfarr))
  function splitoutvals(xs::Vector)
    [out[i] for i in yoids], [out[i] for i in θoids]
  end
  imap(splitoutvals ∘ pgfjl, xgens)
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
  # @grab xabv
  @grab nnettarr
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
  lossarr, n, tabv = δpgfx_ny_arr(f, xabv)
  xgens = [Sampler(()->rand(sz...)) for i = 1:n▸(f)]
  y_θ_gen = x_to_y_θ_gen(pgff, xgens)
  trainpgfnet(lossarr,
              n,
              y_θ_gen,
              tabv,
              TensorFlowTarget.TFTarget,
              TensorFlowTarget.mlp_template)
end