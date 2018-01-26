using Arrows
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
import AlioAnalysis: port_names, pianet, fxgen, trainpianet, Sampler, piareparamnet
import Arrows: NmAbValues, Size

arrs = [TestArrows.sin_arr(),
        TestArrows.xy_plus_x_arr(),
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
  pianetarr = pianet(arr)
  xgens = [Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = fxgen(arr, xgens)

  trainpianet(arr, pianetarr, ygens, xabv, TFTarget, mlp_template;
              cont = data -> data.i < 100) # Only do 100 iterations
end

arrs = [TestArrows.xy_plus_x_arr(),
        TestArrows.abc_arr()]

foreach(arrs) do arr
  println("Testing Preimage attack on $(name(arr)) using Parametric Inverse")
  batch_size = 1
  sz = [batch_size, 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(arr))
  # F -> reparameterized inverse 
  invf = invert(arr, inv, xabv)
  psl = AlioAnalysis.pslnet(invf)
  pianetarr = AlioAnalysis.reparamf(psl, invf)
  lossarr = AlioAnalysis.nlossarr(arr, pianetarr)
  net = first(AlioAnalysis.findnets(lossarr))
  invfinlossar = first(Arrows.findtarrs(lossarr, invf))
  tabv = Arrows.tabvfromxabv(invfinlossar, xabv)
  # Generators
  xgens = [Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = fxgen(arr, xgens)

  # Start training
  AlioAnalysis.optimizenet(lossarr,
    ◂(lossarr, is(ϵ))[1],
    TFTarget,
    mlp_template,
    ingens = ygens,
    xabv = tabv;
    cont = data -> data.i < 100)

#   trainpianet(arr, pianetarr, ygens, xabv, TFTarget, mlp_template;
#               cont = data -> data.i < 100) # Only do 100 iterations
end