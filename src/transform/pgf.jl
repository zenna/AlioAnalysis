"Computes `δ(pgf(x), n(f(x))`"
function δpgfx_ny_arr(f::Arrow, xabv::XAbValues)
  invf = invert(f, inv, xabv)
  @grab tabv = traceprop!(invf, xabv)
  nabv = Arrows.nmfromtabv(tabv, invf)
  n = pslnet(invf)
  lossarr = δny_x_arr(n; nm = :δpgfx_ny_arr)
  lossarr, n, nabv
end

"Iterator of `[f(x), pgf(x)]`  values from `xgenss`: Iterator over inputs "
function x_to_y_θ_gen(pgfarr::Arrow, xgen)
  θprts, yprts = Arrows.partition(is(θp), ◂(pgfarr))
  θoids = out_port_id.(θprts)
  yoids = out_port_id.(yprts)
  pgfjl = il(Arrows.splat(julia(pgfarr)))
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
  tabv = Arrows.tabvfromxabv(nnettarr, xabv)

  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             optimtarget,
             template,
             ingens = y_θ_gen,
             xabv = tabv;
             optimizeargs...)
end