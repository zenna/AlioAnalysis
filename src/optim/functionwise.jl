import TensorFlowTarget: mlp_template, TFTarget

"""Optimize (a function containing) a neural network using TensorFlow

# Arguments
- `carr`:
- `ϵprt`: Error port of arrow you want to minimize

"""
function optimizenet(carr::CompArrow,
                     ϵprt::AbstractPort;
                     template = mlp_template,
                     xabv::XAbValues = TraceAbValues(),
                     ingens = in_port_gen(carr, xabv),
                     optimizeargs...)
  @pre is_valid(carr)
  init_nets!(carr, template, xabv)  # Initialie inner neural networks
  optimize(carr,
           ϵprt,
           ingens,
           TFTarget;
           optimizeargs...)
end


"Initialize a neural network with a composite arrow"
function init_nets!(carr::CompArrow,
                    template = mlp_template,
                    xabv::XAbValues = TraceAbValues())
  @pre issingleton(findnets(carr))

  # Find the network
  nnettarr = first(findnets(carr))   # Find the network and add `func`

  # Compute the input and output sizes
  tabv = traceprop!(carr, xabv)
  insizes = [tabv[tval][:size] for tval in in_trace_values(nnettarr)]
  outsizes = [tabv[tval][:size] for tval in out_trace_values(nnettarr)]
  # Update the template of the network with
  deref(nnettarr).func = args->template(args, insizes, outsizes)
end

"Generate iterators for the inputs of an arrow"
function in_port_gen(carr::CompArrow, xabv::XAbValues)
  @show tabv = traceprop!(carr, xabv)
  insizes = [get(tabv[tval][:size]) for tval in in_trace_values(Arrows.TraceSubArrow(carr))]
  [Sampler{Array}(()->rand(carrinsizes[i]...)) for i = 1:length(▸(carr))]
end
