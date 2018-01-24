"Equivalent to `f` but invokes latest"
il(f) = (args...) -> Base.invokelatest(f, args...)

"Prefix"
pfx(f::Arrow, v::Symbol) = Symbol(name(f), :_, v) # FIXME: Move this elsewhere

function sumsqrerr(fx::Vector, y::Vector)
  @pre length(fx) == length(y)
  sum([δ!(fx[i], y[i]) for i = 1:length(fx)])
end

sumsqrerr(fx, y) = δ!(fx, y)

sumsqrerr(fx::SubPort, y::Vector{SubPort}) = (@pre issingleton(y); δ!(fx, y[1]))
sumsqrerr(fx::Vector{SubPort}, y::SubPort) = (@pre issingleton(x); δ!(fx[1], y))

"Generator of f(x) from `xgens` generator over x"
function fxgen(f::Arrow, xgens)
  @pre n▸(f) == length(xgens)
  fjl = il(julia(f))
  imap(fjl, xgens...)
end

"""
Computes loss: ``|f(n(y)) - y|``
"""
function δny_fx(f::Arrow, n::Arrow, y::Vector)
  yest = f(n(y...)...) # Could be a tuple of single value
  sumsqrerr(yest, y)   #
end

function nlossarr(f::Arrow, n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, [Arrows.props.(▸(n)); [Arrows.Props(false, :δny_fx, Number)]])
  # @grab n
  # @grab f
  @grab lossarr
  δny_fx(f, n, ▹(lossarr)) ⥅ ◃(lossarr, :δny_fx)
  add!(idϵ)(◃(lossarr, :δny_fx))
  @post lossarr is_valid(lossarr)
end

"""
Preimage attack

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
  @pre same(n◂(f), n▸(n), length(gens))
  lossarr = nlossarr(f, n)
  nnettarr = first(findnets(lossarr))
  tabv = TraceAbValues(TraceValue(trace_port(nnettarr, nm)) => abv
                                 for (nm, abv) in xabv)
  optimizenet(lossarr,
             ◂(lossarr, is(ϵ))[1],
             optimtarget,
             template,
             ingens = ygens,
             xabv = tabv;
             optimizeargs...)
end

function pianet(f::CompArrow, xabv::XAbValues)
  # net = UnknownArrow(pfx(f, :pia), ◂(f), ▸(f))
  net = UnknownArrow(pfx(f, :pia), out_port_names(f),
                                   in_port_names(f))
end


# FIXME: Move these elsewhere
port_names(arr) = [nm.name for nm in name.(ports(arr))]
in_port_names(arr) = [nm.name for nm in name.(in_ports(arr))]
out_port_names(arr) = [nm.name for nm in name.(out_ports(arr))]
# test_pia()
