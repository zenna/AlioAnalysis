"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow)
  invcarr = aprx_invert(fwd)
  pslarr = psl(invcarr)
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow)
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

"Execution the run"
function dorun(opt::Dict{Symbol, Any})
  @show fwdarr = opt[:fwdarr]
  invarr = opt[:invarr](fwdarr)
  lossarr = makeloss(invarr, fwdarr, opt[:loss], custϵ=exϵ)
  optimizerun(lossarr, opt[:batch_size])
end

doruin(optpath::String) = dorun(loadopt(optpath))
