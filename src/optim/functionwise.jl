"Optimize (a function containing) a neural network"
function optimizerun(carr::CompArrow,
                     ϵprt=◂(carr, is(ϵ), 1);
                     template=mlp_template,
                     callbacks=[],
                     xabv::XAbValues = TraceAbValues())
  # Find the network and add `func`

  nnettarr = first(findnets(carr))
  @assert is_wired_ok(carr)
  # @show compile(carr)
  println("before tp")
  @show [nm.name for nm in name.(get_ports(carr))]
  @show collect(keys(xabv))
  # export xabv
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
