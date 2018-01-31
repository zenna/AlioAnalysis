"Computes `δ(pgf(x), n(f(x))`"
function δpgfx_ny_arr(f::Arrow, xabv::XAbVals, err = meansqrerr)
  bitlength = 256
  # FIXME, where should this wrap go?
  @grab invf = Arrows.wraponehot(invert(f, inv, xabv), bitlength) 
  @grab tabv = traceprop!(invf, xabv)
  nabv = Arrows.nmfromtabv(tabv, invf)
  n = pslnet(invf)
  lossarr = δny_x_arr(n; nm = :δpgfx_ny_arr, err = err)
  lossarr, n, nabv
end
# FIXME: DRY

"Function that permutes outputof pgf s.t all `y` come before `θ` and order otherwise presvered"
function y_θ_permute(pgfarr::Arrow)
  θprts, yprts = Arrows.partition(is(θp), ◂(pgfarr))
  θoids = out_port_id.(θprts)
  yoids = out_port_id.(yprts)
  function permute(xs::Tuple)
    [[xs[i] for i in yoids]; [xs[i] for i in θoids]]
  end
end

"Function that splits output of int `y` and `θ` and order otherwise presvered"
function y_θ_split(pgfarr::Arrow)
  θprts, yprts = Arrows.partition(is(θp), ◂(pgfarr))
  θoids = out_port_id.(θprts)
  yoids = out_port_id.(yprts)
  function split(xs::Tuple)
    [xs[i] for i in yoids], [xs[i] for i in θoids]
  end
end

"Iterator of `[f(x), pgf(x)]`  values from `xgenss`: Iterator over inputs "
function x_to_y_θ_gen(pgfarr::Arrow, xgen)
  pgfjl = il(Arrows.splat(julia(pgfarr)))
  permute = y_θ_permute(pgfarr)
  @grab ygen = imap(permute ∘ pgfjl, xgen)
end

"""Xo 
Train preimage attack network using pgf

Given forward arrow `f`
- Generate input data `x`
- Generate output data `y = f(x)`
- Learn mapping `net: y -> x` that minimizes argmin(f(net(y)))
"""
function trainpgfnet(lossarr::Arrow,
                     n::Arrow,
                     y_θ_gen,
                     xabv;
                     @req(opt),
                     optimizeargs...)
  # @pre same([n◂(lossarr), n▸(n)]) # What should these be?
  nnettarr = first(Arrows.findtarrs(lossarr, n))
  tabv = Arrows.tabvfromxabv(nnettarr, xabv)
  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             y_θ_gen;
             xabv = tabv,
             opt = opt,
             optimizeargs...)
end