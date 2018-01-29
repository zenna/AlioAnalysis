"""
Optimize (a function containing) a neural network using TensorFlow

# Arguments
- `carr`: computes loss output
- `ϵprt`: Error port of arrow you want to minimize
- `target`: Optimization method, e.g. `TFTarget` for tensorflow
- `ingen`: Iterator of inputs of `carr`
- `optimizeargs`: any argmenets to be passed to optimization
"""
function optimizenet(carr::CompArrow,
                     ϵprt::AbstractPort,
                     target,
                     template;
                     xabv::XAbVals = TraceAbVals(),
                     ingen = in_port_gen(carr, xabv),
                     optimizeargs...)
  @pre is_valid(carr)
  init_nets!(carr, template; xabv=xabv)  # Initialie inner neural networks
  optimize(carr,
           ϵprt,
           ingen,
           target;
           optimizeargs...)
end

"Initialize a neural network with a composite arrow"
function init_net!(carr::CompArrow,
                   nnettarr::TraceSubArrow,
                   template;
                   xabv::XAbVals = TraceAbVals())
  # Compute the input and output sizes
  tabv = traceprop!(carr, xabv)
  # @grab tabv
  insizes = [tabv[tval][:size] for tval in in_trace_values(nnettarr)]
  outsizes = [tabv[tval][:size] for tval in out_trace_values(nnettarr)]
  # Update the template of the network with
  deref(nnettarr).func = args->template(args, insizes, outsizes)
end

"Initialize a neural network with a composite arrow"
function init_nets!(carr::CompArrow,
                    template;
                    xabv::XAbVals = TraceAbVals())
  @pre issingleton(findnets(carr))
  # Find the network
  nnettarr = first(findnets(carr))   # Find the network and add `func`
  init_net!(carr, nnettarr, template; xabv=xabv)
end

"Generate iterators for the inputs of an arrow"
function in_port_gen(carr::CompArrow, xabv::XAbVals)
  @show tabv = traceprop!(carr, xabv)
  insizes = [get(tabv[tval][:size]) for tval in in_trace_values(Arrows.TraceSubArrow(carr))]
  [Sampler{Array}(()->rand(carrinsizes[i]...)) for i = 1:length(▸(carr))]
end
