"Construct a loss right inverse which maps inverse domains of `fwd` "
function genloss(invarr::Arrow, fwd::Arrow, loss; custϵ=ϵ)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  finv = add_sub_arr!(carr, invarr)
  foreach(link_to_parent!, ▹(finv))
  finv◃ = ◃(finv, !is(ϵ))
  finv▹ = ▹(finv, !is(θp))
  fwd◃ = fwd(finv◃...)
  # There MUST be a better way
  if fwd◃ isa SubPort
    fwd◃ = [fwd◃]
  end
  @show δ◃ = [fwd◃[i] + finv▹[i] for i = 1:length(fwd◃)]
  loss◃ = loss(finv◃...)
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃)
  foreach(add!(custϵ) ∘ link_to_parent!, [loss◃])
  foreach(link_to_parent!, ◃(finv, is(ϵ)))
  @assert is_wired_ok(carr)
  carr
end

"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow, nmabv::NmAbValues = NmAbValues())
  sprtabv = SprtAbValues(get_sub_ports(fwd, nm) => abv for (nm, abv) in nmabv)
  invcarr = invert(fwd, inv, sprtabv)
  @assert false
  @show name ∘ get_ports(fwd)
  @show invcarr ∘ get_ports(fwd)
  # Propagate invcarr with the corresponding values from tavc
  pslarr = psl(invcarr)
  # OK
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow, tabv::Dict=TraceAbValues())
  unk = UnknownArrow(Symbol(:nnet_, name(fwd)),
                            length(◂(fwd)), length(▸(fwd)))
end

# Optimize here
function optimizerun(carr::CompArrow, batch_size::Integer, template=mlp_template)
  ϵprt = ◂(carr, is(exϵ), 1)
  # Find the network and add `func`
  nnettarr = first(findnets(carr))
  # Prorblem is that by here we've lost the shape information
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
function stddorun(opt::Dict{Symbol, Any})
  @show fwdarr = opt[:fwdarr]
  invarr = opt[:invarr](fwdarr)
  lossarr = makeloss(invarr, fwdarr, opt[:loss], custϵ=exϵ)
  optimizerun(lossarr, opt[:batch_size])
end

doruin(optpath::String) = dorun(loadopt(optpath))
