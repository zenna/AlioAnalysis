"Minimize naive loss"
function min_naive(carr::Arrow, inputs...)
  naive_loss, ϵid = Arrows.naive_loss(carr, inputs...)
  ϵprt = ◂(naive_loss, ϵid)
  df, update_data = stdrecordcb()
  Arrows.optimize(naive_loss,
                  ▸(naive_loss),
                  ϵprt,
                  rand(2);
                  callbacks = [update_data])
  [df]
end

"Minimize domain loss `arr` using to"
function min_domainϵ(arr::Arrow,
                     outs...;
                     callbacks=[])
  dmloss, ϵid = domain_ovrl(arr)
  invarr = invert(arr)
  aprx_totalize!(invarr)
  idloss = id_loss(arr, invarr)
  init = [outs..., rand(length(▸(dmloss, is(θp))))...]
  optimize(dmloss, ▸(dmloss, is(θp)), ◂(dmloss, ϵid), init;
            callbacks=[stdupdate, domupdate])
  [df, domaindf]
end
