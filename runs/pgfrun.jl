using Arrows
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
using Spec
const AA = AlioAnalysis

"Preimage attack using reparamterized parametric inverse"
function pgftrainrpi(arr, xabv, xgen, opt; optimargs...)
  lossarr, n, xabv = δpgfx_ny_arr(f, xabv, AA.meancrossentropy)
  y_θ_gen = x_to_y_θ_gen(pgff, xgen)
  trainpgfnet(lossarr,
              n,
              y_θ_gen,
              xabv,
              TensorFlowTarget.TFTarget,
              TensorFlowTarget.conv_template;
              optimargs...)
end

function initrun(opt::Dict{Symbol, Any})
  # Generic data frames / callbacks
  df, record = recordrungen(opt[:runname])
  cbs = [record,
         everyn(savedfgen(opt, df), 3),
         everyn(printloss, 5)]

  # Get the data from the bundle
  fwdarr, xabv, xgen = opt[:bundle].fwdarr, opt[:bundle].xabv, opt[:bundle].xgen

  # Setup opt with meta data
  opt[:arrname] = name(opt[:fwdarr])
  opt[:model] = opt[:trainfunc][1]
  lstring = linearstring(opt, :runname,
                              :niters,
                              :model,
                              :batch_size,
                              :arrname,
                              :traindatasize)
  opt[:trainfunc][2](opt[:fwdarr], opt, xgen;
                     callbacks = cbs,
                     logdir = joinpath(opt[:logdir], lstring),
                     cont = data -> data.i < opt[:niters])
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:bundle => AlioZoo.allbundles(),
                     :trainfunc => [(:rpi, pgftrainrpi)],
                     :traindatasize => [1, 2, 5, 40, 500],
                     :batch_size => [1, 32],
                     :niters => 1000)

  println(@__FILE__)
  # Makekwrd non standard
  dispatchruns(optspace,
               @__FILE__,
               initrun;
               toenum=[:fwdarr, :trainfunc, :traindatasize, :batch_size],
               runsbatch=true,
               runnow=false,
               runlocal=false,
               nsamples=1,
               group="iurat",
               ignoreexceptions=false)
end

function main()
  genorrun(genopts, initrun)
end

# genopts()

# How to do generalization test?
main()