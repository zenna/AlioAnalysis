"Get xabv for pslnet from invf"
function pslxabvfrominvf(invf, xabv)
  # Goal is to get shapes of ports of the pslnet
  # All ports names on psl net are same as on invf, and so..
  # We propagate on invf, 
  tabv = traceprop!(invf, xabv)
  nabv = Arrows.nmfromtabv(tabv, invf)
end

# FIXME: This functin no longer has any point
"Computes `δ(pgf(x), n(f(x))`"
function δpgfx_ny_arr(n::Arrow, err = meansqrerr)
  δny_x_arr(n; nm = :δpgfx_ny_arr, err = err)
end

function y_θ_x_(invarr::Arrow)
  θprts, yprts = Arrows.partition(is(θp), ▸(invarr))
  θoids = Arrows.pos_in_in_ports.(θprts)
  yoids = Arrows.pos_in_in_ports.(yprts)
  θoids, yoids
end

"Function that permutes outputof pgf s.t all `y` come before `θ` and order otherwise presvered"
function y_θ_permute(invarr::Arrow)
  θoids, yoids = y_θ_x_(invarr)
  function permute(xs::Tuple)
    [[xs[i] for i in yoids]; [xs[i] for i in θoids]]
  end
end

"Function that splits output of int `y` and `θ` and order otherwise presvered"
function y_θ_split(invarr::Arrow)
  θoids, yoids = y_θ_x_(invarr)
  function split(xs::Tuple)
    [xs[i] for i in yoids], [xs[i] for i in θoids]
  end
end

"Iterator of `[f(x), pgf(x)]`  values from `xgenss`: Iterator over inputs "
function x_to_y_θ_gen(pgfarr::Arrow, invarr::Arrow, xgen)
  pgfjl = il(Arrows.splat(julia(pgfarr)))
  permute = y_θ_permute(invarr)
  @grab ygen = imap(permute ∘ pgfjl, xgen)
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