using Arrows
using Arrows.TestArrows
using Spec
import Arrows: splat, julia, Props, CompArrow, name, props, n▸, δ!
import IterTools: imap
import AlioAnalysis: sumsqrerr
import TensorFlowTarget: mlp_template, optimizenet

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
    xit = AlioAnalysis.Sampler(randofsz)
    yit = AlioAnalysis.Sampler(randofsz)
    zit = imap(carrjl, AlioAnalysis.Sampler(()->(randofsz(), randofsz())))
    optimizenet(lossarr, ◂(lossarr, 1);
                xabv=NmAbVals(:x => AbVals(:size => Size(sz)),
                                :y => AbVals(:size => Size(sz)),
                                :z => AbVals(:size => Size(sz))),
                ingen = [xit, yit, zit],
                cont = data -> data.i < 100)
  end
end