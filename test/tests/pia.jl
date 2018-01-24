using Arrows
using TensorFlowTarget
import TensorFlowTarget: mlp_template
using AlioAnalysis
import AlioAnalysis: port_names, pianet, fxgen, trainpianet
import Arrows: NmAbValues, Size

function test_pia(arr=TestArrows.xy_plus_x_arr(), batch_size=1)
  sz = [batch_size, 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(arr))
  pianetarr = pianet(arr, xabv)
  xgens = [rand(sz...) for i = 1:n▸(arr)]
  ygens = fxgen(arr, xgens)
  trainpianet(arr, pianetarr, ygens, xabv, TFTarget, mlp_template)
end
