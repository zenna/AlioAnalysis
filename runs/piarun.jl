using Arrows
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
import AlioAnalysis: port_sym_names, pianet, fxgen, trainpianet, Sampler
import AlioAnalysis: recordrungen, everyn, savedfgen, printloss, linearstring

import Arrows: NmAbVals, Size
using Spec

# arrs = [TestArrows.sin_arr(),
#         TestArrows.xy_plus_x_arr(),
#         TestArrows.sqrt_check(),
#         TestArrows.abc_arr(),
#         TestArrows.ifelsesimple(),
#         TestArrows.triple_add(),
#         TestArrows.test_two_op(),
#         TestArrows.nested_core()]

# arrs = [TestArrows.xy_plus_x_arr(),
#         TestArrows.abc_arr()]


       
# arrs = [
#         # TestArrows.xy_plus_x_arr(),
#         # TestArrows.abc_arr(),
#         TestArrows.twoxy_plus_x_arr_bcast()]


"Call back for evaluating test data"
function testdatacb(cbdata)
  # The approach I took in pytorch won't work because its
  # expensive to set everything up, dont want to make a new graph

  # Either
  # somehow stop the gradient step, and modify xgen

  # Or compile the loss arrow in julia
end


"Test Generalization of unconstrained net using preimage attack"
function piageneralizetest(arr, opt)
  # Generate all the training data of size opt[:traindatasize], sample from that
  sz = [opt[:batch_size], 1]

  # Training data
  alltraindata = [[rand(sz...)  for i = 1:n▸(arr)] for j = 1:opt[:traindatasize]]
  xgen = Sampler(()->rand(alltraindata))
  xabv = NmAbVals(pnm => AbVals(:size => Size(sz)) for pnm in port_sym_names(arr))
  pianetarr = AlioAnalysis.pianet(arr)
  ygen = fxgen(arr, xgen)

  # Infinite Test Data
  xgentest = Sampler(()->[rand(sz...)  for i = 1:n▸(arr)])
  ygentest = fxgen(arr, xgentest)
  return ygen, ygentest
end

"Preimage attack using unconstrained net"
function piatrainnet(arr, opt; optimargs...)
  sz = [opt[:batch_size], 1]
  xabv = NmAbVals(pnm => AbVals(:size => Size(sz)) for pnm in port_sym_names(arr))
  pianetarr = AlioAnalysis.pianet(arr)
  xgen = [Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygen = fxgen(arr, xgen)
  ygen, ygentest = piageneralizetest(arr, opt)
  trainpianet(arr, pianetarr, ygen, xabv, TFTarget, mlp_template;
              testingen = ygentest, optimargs...)
end

"Preimage attack using reparamterized parametric inverse"
function piatrainrpi(arr, opt; optimargs...)
  sz = [opt[:batch_size], 1]
  xabv = NmAbVals(pnm => AbVals(:size => Size(sz)) for pnm in port_sym_names(arr))
  # F -> reparameterized inverse
  lossarr, tabv = AlioAnalysis.reparamloss(arr, xabv)
  # Generators
  xgen = [AlioAnalysis.Sampler(()->rand(sz...)) for i = 1:n▸(arr)]
  ygen = AlioAnalysis.fxgen(arr, xgen)

  ygen, ygentest = piageneralizetest(arr, opt)

  # Start training
  AlioAnalysis.optimizenet(lossarr,
    ◂(lossarr, is(ϵ))[1],
    TensorFlowTarget.TFTarget,
    TensorFlowTarget.mlp_template;
    ingen = ygen,
    xabv = tabv,
    testingen = ygentest,
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
  lstring = linearstring(opt, :niters, :model, :batch_size, :runname, :arrname,
                              :traindatasize)
  opt[:trainfunc][2](fwdarr, opt;
                     callbacks = cbs,
                     logdir = joinpath(opt[:logdir], lstring),
                     cont = data -> data.i < opt[:niters])
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => TestArrows.plain_arrows(),
                    #  :trainfunc => [(:net, piatrainnet)],
                     :trainfunc => [(:netgeneralize, piatrainrpi),
                                    (:netgeneralize, piatrainnet)],
                    #  :trainfunc => [(:netgeneralize, piatrainnet)],
                    #  :traindatasize => Int.(round(logspace(0, 5, 4))),
                     :traindatasize => [1, 5, 40, 500],
                     :batch_size => [1, 32],
                     :niters => 10000)
  println(@__FILE__)
  # Makekwrd non standard
  train(optspace,
        @__FILE__;
        toenum=[:fwdarr, :trainfunc, :traindatasize],
        runsbatch=true,
        runnow=false,
        runlocal=false,
        dorun=initrun,
        nsamples=3,
        group="zabang",
        ignoreexceptions=false)
end

function main()
  genorrun(genopts, initrun)
end

# genopts()

# How to do generalization test?
main()