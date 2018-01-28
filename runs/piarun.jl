using Arrows
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
import AlioAnalysis: port_sym_names, pianet, fxgen, trainpianet, Sampler
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


       
arrs = [
        # TestArrows.xy_plus_x_arr(),
        # TestArrows.abc_arr(),
        TestArrows.twoxy_plus_x_arr_bcast()]

"Preimage attack using unconstrained net"
function piatrainnet(arr, opt; optimargs...)
  sz = [opt[:batch_size], 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_sym_names(arr))
  pianetarr = AlioAnalysis.pianet(arr)
  xgens = [AlioAnalysis.Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygens = AlioAnalysis.fxgen(arr, xgens)
  AlioAnalysis.trainpianet(arr, pianetarr, ygens, xabv, TensorFlowTarget.TFTarget,
                           TensorFlowTarget.mlp_template;
                           optimargs...)
end

"Preimage attack using reparamterized parametric inverse"
function piatrainrpi(arr, opt; optimargs...)
  sz = [opt[:batch_size], 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_sym_names(arr))
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
    optimargs...)
end

function initrun(opt::Dict{Symbol, Any})
  df, record = recordrungen(opt[:runname])
  cbs = [record,
         everyn(savedfgen(opt, df), 3),
         everyn(printloss, 5)]
  fwdarr = opt[:fwdarr]
  opt[:arrname] = name(opt[:fwdarr])
  opt[:model] = opt[:trainfunc][1]
  lstring = AlioAnalysis.linearstring(opt, :niters, :model, :batch_size, :runname, :arrname)
  opt[:trainfunc][2](fwdarr, opt;
                     callbacks = cbs,
                     logdir = joinpath(opt[:logdir], lstring),
                     cont = data -> data.i < opt[:niters])
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => arrs,
                     :trainfunc => [(:net, piatrainnet)],
                     :batch_size => 32,
                     :niters => 1000)
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
        group="badmanz",
        ignoreexceptions=false)
end

function main()
  genorrun(genopts, initrun)
end

genopts()
# main()