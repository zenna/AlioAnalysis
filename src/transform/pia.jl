# Take a forward arrow

# Generate

"Equivalent to `f` but invokes latest"
il(f) = (args...) -> Base.invokelatest(f, args...)

"Generator of f(x) from `xgens` generator over x"
function fxgen(f::Arrow, xgens)
  @pre n▸(f) == length(xgens)
  fjl = il(julia(f))
  imap(fjl, xgens...)
end


"""
Computes loss: ``|f(n(y)) - y|``
"""
function δny_fx(y, f::Arrow, n::Arrow)
  yest = f(n(y))
  sumsqrerr(yest, y)
end

function nlossarr(f::Arrow, n::Arrow; nm::Symbol=:lossarr)
  lossarr = CompArrow(nm, ◂(f), [:δny_fx])
  ▹lossarr  = ⬨(lossarr)
  δny_fx(▹lossarr, f, n) ⥅ ◃(lossarr, :δny_fx)
  @post lossarr is_valid(lossarr)
end

"""
Preimage attack

Given forward arrow `f`
- Generate input data `x`
- Generate output data `y = f(x)`
- Learn mapping `net: y -> x` that minimizes argmin(f(net(y)))
"""
function preimgattack(f::Arrow, n::Arrow, ygens)
  @pre same(n◂(f), n▸(n), length(gens))
  lossarr = nlossarr(f, n)

  optimizenet(lossarr, ◂(lossarr, 1);
              xabv=NmAbValues(:x => AbValues(:size => Size(sz)),
                              :y => AbValues(:size => Size(sz)),
                              :z => AbValues(:size => Size(sz))),

  # Make generators for precompute y values offline


  # Optimize

end

"Prefix"
pfx(f::Arrow, v::Symbol) = Symbol(name(f), :_, v)

function pia(carr::CompArrow xabv::XAbValues)
  net = UnknownArrow(pfx(f, :pia), ◂(f), ▸(f))
end

function test_pia(arr=TestArrows.xy_plus_x_arr())
  sz = [batch_size, 1]
  xabv=NmAbValues(:x => AbValues(:size => Size(sz)),
                  :y => AbValues(:size => Size(sz)),
                  :z => AbValues(:size => Size(sz)))
  pia(arr, xabv)
end

# test_pia()
