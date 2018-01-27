"""
Compute loss: |n(y) - x|
"""
function δny_x(n::Arrow, y::Vector, x::Vector)
  @pre n▸(n) == length(y)
  @pre n◂(n) == length(x)
  xest = n(y...)
  sumsqrerr(x, xest)
end

"""(Partial) inverse of `vcat`

```jldoctest
julia> invvcat([1,2,3,4,5,6], 3)
([1, 2, 3], [4, 5, 6])
```
"""
invvcat(xs, i::Integer) = (@pre 1 < i < length(xs); (xs[1:i], xs[i+1:end]))

"Arrow `lossarr: x, y -> δ(f(n(y)), y)`"
function δny_x_arr(n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, [▸(n); ◂(n)], [:δny_x])
  y, x = invvcat(▸(lossarr), n▸(n))
  δny_x(n, y) ⥅ ◃(lossarr, :δny_x)
  add!(ϵ)(◃(lossarr, :δny_x))
  @post lossarr is_valid(lossarr)
end