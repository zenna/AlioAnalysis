# TODO:
# 1. Set the right loss term
# 2. Template?
# 4. prodsample
# 5. Command line - Pickle the options, save, and pass path to ocmmandline

"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow)
  invcarr = aprx_invert(fwd)
  pslarr = psl(invcarr)
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow,)
  unk = UnknownArrow(Symbol(:nnet_, name(fwd)),
                            length(◂(fwd)), length(▸(fwd)))
end


function optimizerun(carr::CompArrow, batch_size::Integer, template=mlp_template)
  ϵprt = ◂(carr, is(exϵ), 1)
  # Find the network and add `func`
  nnettarr = first(findnets(carr))
  insizes = [Size([batch_size, 1]) for i = 1:length(▸(deref(nnettarr)))]
  outsizes = [Size([batch_size, 1]) for i = 1:length(◂(deref(nnettarr)))]
  deref(nnettarr).func = args->mlp_template(args, insizes, outsizes)

  # Setup callbacks
  df, std_cb = savedata()
  callbacks = [std_cb]

  # Optimize
  optimize(carr,
           ϵprt,
           [Arrows.Sampler{Array}(()->rand(batch_size, 1)) for i = 1:length(▸(carr))],
           TFTarget;
           cont=data -> data.i < 400,
           callbacks=callbacks)
  [df]
end

function dorun(opt)
  fwdarr = opt.fwdarr
  invarr = opt.invarr(fwdarr, opt.loss)
  lossarr = makeloss(invarr, opt.loss, custϵ=exϵ)
  optimizerun(lossarr, opt.batch_size)
end

using FileIO
"Run to generate data for initialization comparison"
function initrun()
  optspace = Dict(:fwdarr => [TestArrows.xy_plus_x_arr(), TestArrows.abc_arr()],
                  :batch_size => [16, 32, 64],
                  :invarr => [netpi, invnet],
                  :loss => +,
                  :check => rand)
  for (i, opt) in enumerate(prodsample(optspace, [:fwdarr, :batch_size], [:check], 2))
    saveopt("/Users/zenna/example$i.opt", opt)
    # dorun(opt)
  end
end

