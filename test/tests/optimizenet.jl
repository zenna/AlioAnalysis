using Arrows
using Arrows.TestArrows
using Spec
import Arrows: splat, julia, Props, CompArrow, name, props, n▸, δ!
using Iterators
import AlioAnalysis: optimizenet
import TensorFlowTarget: mlp_template

function sumsqrerr(fx::Vector, y::Vector)
  @pre length(fx) == length(y)
  sum([δ!(fx[i], y[i]) for i = 1:length(fx)])
end

sumsqrerr(fx, y) = δ!(fx, y)

"Compute f(x) - y"
function fx_y(x, y, f::Arrow)
  yest = f(x...) # y estimates
  sumsqrerr(yest, y)
end

"Test Optimization"
function test_optimizenet()
  with_pre() do
    batch_size = 1
    sz = [batch_size, 1]
    randofsz = () -> rand(sz...)

    carr = TestArrows.xy_plus_x_arr()
    carrjl_ = splat(julia(carr))
    carrjl = (args...) -> Base.invokelatest(carrjl_, args...)
    net = UnknownArrow(:nnet, [:x, :y], [:z])
    lossarr = CompArrow(:whoknows, [:x, :y, :z], [:loss])
    x, y, z, loss = ⬨(lossarr)
    fx_y([x, y], z, net) ⥅ loss
    @assert is_valid(lossarr)
    # Generate data
    xit = Arrows.Sampler(randofsz)
    yit = Arrows.Sampler(randofsz)
    zit = imap(carrjl, Arrows.Sampler(()->(randofsz(), randofsz())))
    optimizenet(lossarr, ◂(lossarr, 1);
                xabv=NmAbValues(:x => AbValues(:size => Size(sz)),
                                :y => AbValues(:size => Size(sz)),
                                :z => AbValues(:size => Size(sz))),
                ingens = [xit, yit, zit],
                cont = data -> data.i < 100)
  end
end

# test_optimizenet()
function test_optimizenet_pgf()
  pgfcarr = pgf(carr)
  pgfjl = julia(pgfcarr)
  yit = imap(carrjl, xit)
  θit = imap(pgfjl, xit)
end


# What's painful here?
# 1) passing around these xabv values
# 2) no types to generate values from
# 3)
