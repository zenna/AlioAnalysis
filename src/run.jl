plus(x::SubPort) = x
plus(xs::SubPort...) = +(xs...)

"Construct a loss right inverse which maps inverse domains of `fwd` "
function genloss(invarr::Arrow, fwd::Arrow, loss)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  finv = add_sub_arr!(carr, invarr)
  # idϵ
  foreach(link_to_parent!, ▹(finv))
  finv◃ = ◃(finv, !is(ϵ))
  finv▹ = ▹(finv, !is(θp))
  fwd◃ = fwd(finv◃...)
  # There MUST be a better way
  if fwd◃ isa SubPort
    fwd◃ = [fwd◃]
  end

  # Sum the losses from each output
  δ◃s = [fwd◃[i] + finv▹[i] for i = 1:length(fwd◃)]
  δtot◃ = plus(δ◃s...)
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃s)

  # Any other loss
  loss◃ = loss(finv◃...)
  (add!(ϵ) ∘ link_to_parent!)(loss◃)

  # Total loss to minimize
  tomin◃ = loss◃ + δtot◃
  link_to_parent!(tomin◃)

  # Link every output output to parent
  foreach(link_to_parent!, ◃(finv))
  @assert is_wired_ok(carr)
  carr
end

"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow, nmabv::NmAbValues = NmAbValues())
  sprtabv = SprtAbValues(⬨(fwd, nm) => abv for (nm, abv) in nmabv)
  invcarr = aprx_invert(fwd, inv, sprtabv)
  pslarr = psl(invcarr)
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow, tabv::Dict=TraceAbValues())
  @show unk = UnknownArrow(Symbol(:invnet_, name(fwd)),
                           [name(prt).name for prt in ◂(fwd)],
                           [name(prt).name for prt in ▸(fwd)])
end

# Optimize here
function optimizerun(carr::CompArrow,
                     ϵprt=◂(carr, is(ϵ), 1);
                     template=mlp_template,
                     callbacks=[],
                     xabv::XAbValues = TraceAbValues())
  # Find the network and add `func`

  nnettarr = first(findnets(carr))
  tabv = traceprop!(carr, xabv)
  # @show values(tabv)
  insizes = [tabv[tval][:size] for tval in in_trace_values(nnettarr)]
  outsizes = [tabv[tval][:size] for tval in out_trace_values(nnettarr)]
  deref(nnettarr).func = args->mlp_template(args, insizes, outsizes)

  carrinsizes = [get(tabv[tval][:size]) for tval in in_trace_values(Arrows.TraceSubArrow(carr))]
  gens = [Sampler{Array}(()->rand(carrinsizes[i]...)) for i = 1:length(▸(carr))]

  # Setup callbacks
  df, std_cb = savedata()
  callbacks = [std_cb]

  # Optimize
  optimize(carr,
           ϵprt,
           gens,
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
