function benchmarkarrs()
  [TestArrows.xy_plus_x_arr(),
  TestArrows.abc_arr(),
  TestArrows.weird_arr()]
end

function initrun(opt::Dict{Symbol, Any})
  println("Starting init")
  df, record = recordrungen()
  cbs = [record,
         everyn(savedfgen("rundata", joinpath(opt[:logdir], "rundata.jld2"), df), 3)]
  everyn(printloss, 5)
  # nmabv = NmAbValues(nm => AbValues(:size => val) for (nm, val) in szs)
  fwdarr = opt[:fwdarr]
  @grab fwdarr
  xvals = [rand() for i = 1:length(▸(fwdarr))] # FIXME: assumes scalar
  yvals = fwdarr(xvals...)
  opt[:minf](fwdarr, yvals...; callbacks=cbs)
  println("Finishing init")
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => benchmarkarrs(),
                    #  :minf => [min_domainϵ])
                     :minf => [min_naive, min_domainϵ])
  println(@__FILE__)
  # Makekwrd non standard
  train(optspace,
        @__FILE__;
        toenum=[:fwdarr, :minf],
        runnow=true,
        runlocal=false,
        dorun=initrun,
        nsamples=200,
        group="domvsid")
  end

function main()
  genorrun(genopts, initrun)
end

genopts()
# main()
