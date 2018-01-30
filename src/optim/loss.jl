## Loss functions
"`δ(a, b)`"
l2norm(a::SubPort, b::SubPort) = sqrt(sqr(a - b)) # TODO: Make type specific

"Cross entropy loss"
function cross_entropy(fx::SubPort, y::SubPort)
  ok = -reduce_sum(fx * log(y), axis=2)
  reduce_mean(ok)
end

"`f(δ(x[i], y[i]))` or all `i`"
function accumerror(f, δ, xs::Vector, ys::Vector)
  @pre length(xs) == length(ys)
  f(map(δ, xs, ys))
end

accumerror(f, δ, fx::SubPort, y::SubPort) = δ(fx, y)
accumerror(f, δ, fx::SubPort, y::Vector{SubPort}) = (@pre issingleton(y); δ(fx, y[1]))
accumerror(f, δ, fx::Vector{SubPort}, y::SubPort) = (@pre issingleton(fx); δ(fx[1], y))

"Summed square error"
sumsqrerr(xs, ys) = accumerror(sum, l2norm, xs, ys)

"Mean Square Error"
meansqrerr(xs, ys) = accumerror(mean, l2norm, xs, ys)

"Mean cross entropy"
meancrossentropy(xs, ys) = accumerror(mean, cross_entropy, xs, ys)