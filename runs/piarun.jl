using Arrows
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
import AlioAnalysis: port_names, pianet, fxgen, trainpianet, Sampler
import AlioAnalysis: recordrungen, everyn, savedfgen, printloss

import Arrows: NmAbValues, Size
using Spec

arrs = [TestArrows.sin_arr(),
        TestArrows.xy_plus_x_arr(),
        TestArrows.sqrt_check(),
        TestArrows.abc_arr(),
        TestArrows.ifelsesimple(),
        TestArrows.triple_add(),
        TestArrows.test_two_op(),
        TestArrows.nested_core()]

arrs = [TestArrows.xy_plus_x_arr(),
        TestArrows.abc_arr()]

"Preimage attack using unconstrained net"
function piatrainnet(arr, opt)
  sz = [opt[:batch_size], 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(arr))
  pianetarr = AlioAnalysis.pianet(arr)
  xgens = [AlioAnalysis.Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = AlioAnalysis.fxgen(arr, xgens)
  AlioAnalysis.trainpianet(arr, pianetarr, ygens, xabv, TensorFlowTarget.TFTarget,
                           TensorFlowTarget.mlp_template;
                           cont = data -> data.i < 1000) # Only do 100 iterations
end

"Preimage attack using reparamterized parametric inverse"
function piatrainrpi(arr, opt)
  sz = [opt[:batch_size], 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(arr))
  # F -> reparameterized inverse
  lossarr, tabv = AlioAnalysis.reparamloss(arr, xabv)
  # Generators
  xgens = [AlioAnalysis.Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = AlioAnalysis.fxgen(arr, xgens)
  # Start training
  AlioAnalysis.optimizenet(lossarr,
    ◂(lossarr, is(ϵ))[1],
    TensorFlowTarget.TFTarget,
    TensorFlowTarget.mlp_template,
    ingens = ygens,
    xabv = tabv;
    cont = data -> data.i < 100)
end

function initrun(opt::Dict{Symbol, Any})
  df, record = recordrungen(opt[:runname])
  cbs = [record,
         everyn(savedfgen(opt, df), 3),
         everyn(printloss, 5)]
  fwdarr = opt[:fwdarr]
  opt[:trainfunc](fwdarr, opt)
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => arrs,
                     :trainfunc => [piatrainnet, piatrainnet],
                     :batch_size => 32)
  println(@__FILE__)
  # Makekwrd non standard
  train(optspace,
        @__FILE__;
        toenum=[:fwdarr, :trainfunc],
        runsbatch=false,
        runnow=true,
        runlocal=false,
        dorun=initrun,
        nsamples=1,
        group="test_analysis_pia_2",
        ignoreexceptions=false)
end

# genopts()