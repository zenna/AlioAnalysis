using Arrows
using TensorFlowTarget
import TensorFlowTarget: mlp_template
using AlioAnalysis
import AlioAnalysis: port_names, pianet, fxgen, trainpianet, Sampler
import Arrows: NmAbValues, Size

arrs = [TestArrows.sin_arr(),
        TestArrows.xy_plus_x_arr(),
        TestArrows.twoxy_plus_x_arr(),
        TestArrows.sqrt_check(),
        TestArrows.abc_arr(),
        TestArrows.ifelsesimple(),
        TestArrows.triple_add(),
        TestArrows.test_two_op(),
        TestArrows.nested_core()]

foreach(arrs) do arr
  println("Testing Preimage attack on $(name(arr))")
  batch_size = 1
  sz = [batch_size, 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(arr))
  pianetarr = pianet(arr, xabv)
  xgens = [Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = fxgen(arr, xgens)

  trainpianet(arr, pianetarr, ygens, xabv, TFTarget, mlp_template;
              cont = data -> data.i < 100) # Only do 100 iterations
end
