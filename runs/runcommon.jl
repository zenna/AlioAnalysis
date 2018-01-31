"Run initializer used by many runs"
function commoninitrun(opt::Dict{Symbol, Any})
  # Generic data frames / callbacks
  df, record = recordrungen(opt[:runname])
  cbs = [record,
         everyn(savedfgen(opt, df), 3),
         everyn(printloss, 5)]
         
  # Get the data from the bundle
  bundle = opt[:bundlegen](; opt...)

  # Setup opt with meta data
  opt[:arrname] = name(bundle.fwdarr)
  opt[:model] = opt[:trainfunc][1]
  lstring = linearstring(opt, :runname,
                              :niters,
                              :model,
                              :batch_size,
                              :arrname,
                              :traindatasize)
  opt[:trainfunc][2](bundle;
                     opt = opt,
                     callbacks = cbs,
                     logdir = joinpath(opt[:logdir], lstring),
                     cont = data -> data.i < opt[:niters])
end