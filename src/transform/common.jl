"Equivalent to `f` but invokes latest"
il(f) = (args...) -> Base.invokelatest(f, args...)

"""
Iterator of `f(x)` from `xgen` generator over `x`

```jldoctest
julia> arr = Arrows.TestArrows.xy_plus_x_arr()
julia> fmap = fxgen(arr, [(rand() for i = 1:10), (rand() for i = 1:10)])
```
"""
function fxgen(f::Arrow, xgen)
  @pre n▸(f) == length(first(xgen)) # XXX: is this safe?
  fjl = il(Arrows.splat(julia(f)))
  if n◂(f) == 1
    imap(tuple ∘ fjl, xgen)
  else 
    imap(fjl, xgen)
  end
end

fsplat(f, arg::SubPort) = f(arg)
fsplat(f, args::Vector{SubPort}) = f(args...)
