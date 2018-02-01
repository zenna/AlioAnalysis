using Arrows
using AlioZoo
const AZ = AlioZoo
import TensorFlowTarget: mlp_template, TFTarget
using AlioAnalysis
const AA = AlioAnalysis
using Spec
using TensorFlow
const tf = TensorFlow

include("runcommon.jl")

# What's still wrong
# Template
# call to dispatch runs should take cmdline args
# Continue when no improvement

function train_hypers()
  @show lr = rand([0.0001, 0.00001])
  optimizer = tf.train.AdamOptimizer(lr)
end

"Preimage attack using reparamterized parametric inverse"
function pgftrainrpi(bundle; @req(opt), optimargs...)
  @grab n = AA.pslnet(bundle.invf)
  @grab bundle
  xabv = :pslxabv ∈ keys(bundle) ? bundle.pslxabv : AA.pslxabvfrominv(invf, bundle.xabv)
  lossarr = AA.δpgfx_ny_arr(n, opt[:loss])
  @grab y_θ_gen = AA.x_to_y_θ_gen(bundle.pgff, bundle.invf, bundle.gen)
  AA.trainpgfnet(lossarr,
                 n,
                 y_θ_gen,
                 xabv;
                 opt = opt,
                 optimargs...)
end

"Generate data for initialization comparison"
function genopts()
  optspace = Options(:bundlegen => AZ.allbundlegens,
                     :trainfunc => [(:rpi, pgftrainrpi)],
                     :traindatasize => [1, 2, 5, 40, 500],
                    #  :loss => [AA.meancrossentropy],
                     :loss => AA.meansqrerr,
                     :batch_size => [32, 64],
                     :target => TensorFlowTarget.TFTarget,
                     :template => TensorFlowTarget.conv_template,
                     :niters => 100000,
                     :hyper_gen => train_hypers,
                     :netparams => TensorFlowTarget.rand_convnet_hypers)

  println(@__FILE__)
  dispatchruns(optspace,
               @__FILE__,
               commoninitrun;
               toenum=[:bundlegen, :trainfunc, :traindatasize],
               tosample=[:netparams, :batch_size],
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
