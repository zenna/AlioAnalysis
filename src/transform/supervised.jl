"""
Compute loss: |n(y) - x|
"""
function δny_x(n::Arrow, y::Vector, x::Vector)
  @pre n▸(n) == length(y)
  @pre n◂(n) == length(x)
  xest = n(y...)
  sumsqrerr(x, xest)
end

"Given `n : y -> x`, Arrow `lossarr: y, x -> δ(f(n(y)), y)`"
function δny_x_arr(n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, port_sym_name.([▸(n); ◂(n)]), [:δny_x])
  y, x = Arrows.invvcat(▹(lossarr), n▸(n))
  δny_x(n, y, x) ⥅ ◃(lossarr, :δny_x)
  add!(ϵ)(◃(lossarr, :δny_x))
  @post lossarr is_valid(lossarr)
end