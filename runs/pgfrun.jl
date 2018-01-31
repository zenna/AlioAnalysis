using Arrows
using AlioZoo
const AZ = AlioZoo
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
const AA = AlioAnalysis
using Spec

include("runcommon.jl")

# What's still wrong
# Template
# call to dispatch runs should take cmdline args
# Continue when no improvement

"Preimage attack using reparamterized parametric inverse"
function pgftrainrpi(bundle; optimargs...)
  @grab bundle
  lossarr, n, xabv = AA.δpgfx_ny_arr(bundle.fwdarr, bundle.xabv, AA.meancrossentropy)
  y_θ_gen = AA.x_to_y_θ_gen(bundle.pgff, bundle.gen)
  AA.trainpgfnet(lossarr,
                 n,
                 y_θ_gen,
                 xabv,
                 TensorFlowTarget.TFTarget,
                 TensorFlowTarget.conv_template;
                 optimargs...)
end

"Generate data for initialization comparison"
function genopts()
  optspace = Options(:bundlegen => AZ.allbundlegens,
                     :trainfunc => [(:rpi, pgftrainrpi)],
                     :traindatasize => [1, 2, 5, 40, 500],
                     :batch_size => [1, 32],
                     :niters => 1000)

  println(@__FILE__)
  dispatchruns(optspace,
               @__FILE__,
               commoninitrun;
               toenum=[:bundlegen, :trainfunc, :traindatasize, :batch_size],
               runsbatch=false,
               runnow=true,
               runlocal=false,
               nsamples=1,
               group="iurat",
               ignoreexceptions=false)
end

function main()
  genorrun(genopts, commoninitrun)
end

genopts()

# How to do generalization test?
# main()