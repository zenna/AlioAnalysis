"Find `x` such that `fwd(x) = yvals`"
function min_naive(fwd::Arrow,
                   yvals...;
                   callbacks=[],
                   opt=Dict(),
                   xabv::XAbVals = Arrows.TraceAbVals(),
                   pgfvals = [])
  naive_loss, ϵid = Arrows.naive_loss(fwd, yvals)
  ϵprt = ◂(naive_loss, ϵid)
  init = rand(length(▸(naive_loss))) # FIXME: assumes scalar
  optimize(naive_loss,
           ▸(naive_loss),
           ϵprt,
           init;
           callbacks = callbacks)
end

"Construct inverse of `fwd` and minimize domain loss"
function min_domϵ(fwd::Arrow,
                 outs...;
                 callbacks=[],
                 opt=Dict(),
                 pgfvals = [])
   min_superϵ(fwd, :totdomϵ, outs...; callbacks=callbacks, opt=opt, pgfvals = pgfvals)
end

"Construct inverse of `fwd` and minimize domain loss"
function min_idϵ(fwd::Arrow,
                 outs...;
                 callbacks=[],
                 opt=Dict(),
                 pgfvals = [])
   min_superϵ(fwd, :idtotal, outs...; callbacks=callbacks, opt=opt, pgfvals=pgfvals)
end

"Construct inverse of `fwd` and minimize domain loss"
function min_both(fwd::Arrow,
                 outs...;
                 callbacks=[],
                 opt=Dict(),
                 pgfvals = [])
   min_superϵ(fwd, :both, outs...; callbacks=callbacks, opt=opt, pgfvals = pgfvals)
end

"Construct inverse of `fwd` and minimize domain loss"
function min_superϵ(fwd::Arrow,
                    tag::Symbol,
                    outs...;
                    callbacks=[],
                    opt=Dict(),
                    pgfvals=pgfvals)
  @show opt
  res = Arrows.superloss(fwd)
  invarr = res[:invcarr]
  ϵprt = res[tag]
  df = DataFrame(idtotal = Float64[], domtotal = Float64[], domall = Vector[])
  function mycallback(data)
    @show idtotal = data.output[pos_in_out_ports(res[:idtotal])]
    domtotal = data.output[pos_in_out_ports(res[:totdomϵ])]
    domids = [pos_in_out_ports(prt) for prt in ◂(invarr, is(idϵ)) if prt != res[:totdomϵ]]
    domlosses = [data.output[domid] for domid in domids]
    push!(df, [idtotal, domtotal, domlosses])
  end
  savedf = everyn(savedfgen("pilosses", joinpath(opt[:logdir], "pilosses.jld2"), df), 3)
  # @show pgfvals
  # @assert false
  # initthetas = rand(length(▸(invarr, is(θp))))
  initthetas = pgfvals[13:end]
  @show length(initthetas)
  @show length(initthetas)
  init = [outs..., initthetas...]
  optimize(invarr,
           ▸(invarr, is(θp)),
           ϵprt,
           init;
           callbacks = vcat(callbacks, mycallback, savedf))
end
