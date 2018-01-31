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
                     ingen;
                     @req(opt),
                     xabv::XAbVals = TraceAbVals(),
                     optimizeargs...)
  @pre is_valid(carr)
  init_nets!(carr; opt=opt, xabv=xabv)  # Initialie inner neural networks
  optimize(carr,
           ϵprt,
           ingen,
           opt[:target];
           opt=opt,
           optimizeargs...)
end

"Initialize a neural network with a composite arrow"
function init_net!(carr::CompArrow,
                   nnettarr::TraceSubArrow;
                   @req(opt),
                   xabv::XAbVals = TraceAbVals(),
                   optimizeargs...)
  # Compute the input and output sizes
  tabv = traceprop!(carr, xabv)
  insizes = [tabv[tval][:size] for tval in in_trace_values(nnettarr)]
  outsizes = [tabv[tval][:size] for tval in out_trace_values(nnettarr)]
  # Update the template of the network with
  template = opt[:template]
  netparams = opt[:netparams]
  deref(nnettarr).func = args->template(args, insizes, outsizes; netparams...)
end

"Initialize a neural network with a composite arrow"
function init_nets!(carr::CompArrow;
                    @req(opt),
                    xabv::XAbVals = TraceAbVals(),
                    optimizeargs...)
  @pre issingleton(findnets(carr))
  # Find the network
  nnettarr = first(findnets(carr))   # Find the network and add `func`
  init_net!(carr, nnettarr; opt = opt, xabv=xabv, optimizeargs...)
end