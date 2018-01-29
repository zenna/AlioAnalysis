# (i) PreImage attack (pia): given `y`, find any `x` s.t. `f(x) = y
# (ii) Amortized PreImage attack, find `n: Y -> X`, s.t. forall `y`, (i) holds

"""
Computes loss: ``|f(n(y)) - y|``
"""
function δfny_y(f::Arrow, n::Arrow, y::Vector)
  xest = n(y...)
  yest = fsplat(f, xest)
  sumsqrerr(yest, y)   #
end

"Arrow `lossarr: x, y -> δ(f(n(y)), y)`"
function δfny_y_arr(f::Arrow, n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, [Arrows.props.(▸(n)); [Arrows.Props(false, :δfny_y, Number)]])
  δfny_y(f, n, ▹(lossarr)) ⥅ ◃(lossarr, :δfny_y)
  add!(idϵ)(◃(lossarr, :δfny_y))
  @post lossarr is_valid(lossarr)
end

"""
Train preimage attack network

Given forward arrow `f`
- Generate input data `x`
- Generate output data `y = f(x)`
- Learn mapping `net: y -> x` that minimizes argmin(f(net(y)))
"""
function trainpianet(f::Arrow,
                     n::Arrow,
                     ygens,
                     xabv,
                     optimtarget,
                     template;
                     optimizeargs...)
  @pre same([n◂(f), n▸(n)])
  lossarr = δfny_y_arr(f, n)
  nnettarr = first(findnets(lossarr))
  tabv = Arrows.tabvfromxabv(nnettarr, xabv)
  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             optimtarget,
             template,
             ingen = ygens,
             xabv = tabv;
             optimizeargs...)
end

"Preimage attack network to invert `f`"
function pianet(f::Arrow)
  net = UnknownArrow(pfx(f, :pia), out_port_sym_names(f),
                                   in_port_sym_names(f))
end