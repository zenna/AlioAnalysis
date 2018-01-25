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
  
function initrun(opt::Dict{Symbol, Any})
  df, record = recordrungen(opt[:runname])
  cbs = [record,
         everyn(savedfgen(opt, df), 3),
         everyn(printloss, 5)]
  fwdarr = opt[:fwdarr]

  # TODO: Get Types/Sizes of frward arrow
  # Generate input (to forward arrow) data
  batch_size = 1
  sz = [batch_size, 1]
  xabv = NmAbValues(pnm => AbValues(:size => Size(sz)) for pnm in port_names(fwdarr))

  # Generates PIA net
  pianetarr = pianet(fwdarr, xabv)
  xgens = [Sampler(()->rand(sz...)) for i = 1:n▸(fwdarr)]
  ygens = fxgen(fwdarr, xgens)

  trainpianet(fwdarr, pianetarr, ygens, xabv, TFTarget, mlp_template;
              cont = data -> data.i < 100,
              callbacks = cbs) # Only do 100 iterations
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => arrs)
  println(@__FILE__)
  # Makekwrd non standard
  train(optspace,
        @__FILE__;
        toenum=[:fwdarr],
        runsbatch=false,
        runnow=true,
        runlocal=false,
        dorun=initrun,
        nsamples=1,
        group="test_analysis_pia_2",
        ignoreexceptions=false)
end

# genopts()