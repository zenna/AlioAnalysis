"Equivalent to `f` but invokes latest"
il(f) = (args...) -> Base.invokelatest(f, args...)

"Prefix"
pfx(f::Arrow, v::Symbol) = Symbol(name(f), :_, v) # FIXME: Move this elsewhere

function sumsqrerr(fx::Vector, y::Vector)
  @pre length(fx) == length(y)
  sum([δ!(fx[i], y[i]) for i = 1:length(fx)])
end

sumsqrerr(fx::SubPort, y::SubPort) = δ!(fx, y)
sumsqrerr(fx::SubPort, y::Vector{SubPort}) = (@pre issingleton(y); δ!(fx, y[1]))
sumsqrerr(fx::Vector{SubPort}, y::SubPort) = (@pre issingleton(fx); δ!(fx[1], y))

"""
Iterator of f(x) from `xgens` generator over x

```jldoctest
julia> arr = Arrows.TestArrows.xy_plus_x_arr()
julia> fmap = fxgen(arr, [(rand() for i = 1:10), (rand() for i = 1:10)])
```
"""
function fxgen(f::Arrow, xgens)
  @pre n▸(f) == length(xgens)
  fjl = il(julia(f))
  if n◂(f) == 1
    imap(tuple ∘ fjl, xgens...)
  else 
    imap(fjl, xgens...)
  end
end

fsplat(f, arg::SubPort) = f(arg)
fsplat(f, args::Vector{SubPort}) = f(args...)

"""
Computes loss: ``|f(n(y)) - y|``
"""
function δny_fx(f::Arrow, n::Arrow, y::Vector)
  xest = n(y...)
  yest = fsplat(f, xest)
  sumsqrerr(yest, y)   #
end

function nlossarr(f::Arrow, n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, [Arrows.props.(▸(n)); [Arrows.Props(false, :δny_fx, Number)]])
  δny_fx(f, n, ▹(lossarr)) ⥅ ◃(lossarr, :δny_fx)
  add!(idϵ)(◃(lossarr, :δny_fx))
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
                     template; optimizeargs...)
  @pre same([n◂(f), n▸(n)])
  lossarr = nlossarr(f, n)
  nnettarr = first(findnets(lossarr))
  tabv = Arrows.tabvfromxabv(nnettarr, xabv)
  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             optimtarget,
             template,
             ingens = ygens,
             xabv = tabv;
             optimizeargs...)
end

"Preimage attack network to invert `f`"
function pianet(f::Arrow)
  net = UnknownArrow(pfx(f, :pia), out_port_names(f),
                                   in_port_names(f))
end