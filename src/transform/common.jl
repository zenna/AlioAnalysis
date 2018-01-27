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
