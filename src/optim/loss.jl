## Loss functions
"`δ(a, b)`"
l2norm(a::SubPort, b::SubPort) = sqrt(sqr(a - b)) # TODO: Make type specific

"Cross entropy loss"
function cross_entropy(fx, y)
  reduce_mean(-reduce_sum(fx .* log(y), axis=[2]))
end

"`f(δ(x[i], y[i]))` or all `i`"
function fsqrerror(f, δ, xs::Vector, ys::Vector)
  @pre length(xs) == length(ys)
  f([δ(xs[i], ys[i]) for i = 1:length(xs)])
end

fsqrerror(f, δ, fx::SubPort, y::SubPort) = δ(fx, y)
fsqrerror(f, δ, fx::SubPort, y::Vector{SubPort}) = (@pre issingleton(y); δ(fx, y[1]))
fsqrerror(f, δ, fx::Vector{SubPort}, y::SubPort) = (@pre issingleton(fx); δ(fx[1], y))

"Summed square error"
sumsqrerr(xs, ys) = fsqrerror(sum, l2norm, xs, ys)

"Mean Square Error"
meansqrerr(xs, ys) = fsqrerror(mean, l2norm, xs, ys)