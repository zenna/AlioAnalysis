"Find `x` such that `fwd(x) = yvals`"
function min_naive(fwd::Arrow,
                   yvals...;
                   callbacks=[],
                   xabv::XAbValues = Arrows.TraceAbValues())
  naive_loss, ϵid = Arrows.naive_loss(fwd, yvals...)
  ϵprt = ◂(naive_loss, ϵid)
  init = rand(length(▸(naive_loss))) # FIXME: assumes scalar
  optimize(naive_loss,
           ▸(naive_loss),
           ϵprt,
           init;
           callbacks = callbacks)
end

"Construct inverse of `fwd` and minimize domain loss"
function min_domainϵ(fwd::Arrow,
                     outs...;
                     callbacks=[])
  dmloss, ϵid = domain_ovrl(fwd)
  invarr = invert(fwd)
  aprx_totalize!(invarr)
  idloss = id_loss(fwd, invarr)
  init = [outs..., rand(length(▸(dmloss, is(θp))))...]
  optimize(dmloss,
           ▸(dmloss, is(θp)),
           ◂(dmloss, ϵid),
           init;
           callbacks=[])
end
