using Arrows
using AlioZoo
using AlioAnalysis
import AlioAnalysis: min_naive, recordrungen, min_domainϵ, everyn, savedfgen, printloss


function initrun(opt::Dict{Symbol, Any})
  df, record = recordrungen()
  cbs = [record,
         everyn(savedfgen("rundata", joinpath(opt[:logdir], "rundata.jld2"), df), 3),
         everyn(printloss, 5)]
  # nmabv = NmAbValues(nm => AbValues(:size => val) for (nm, val) in szs)
  fwdarr = opt[:fwdarr]
  @grab fwdarr
  xvals = [rand() for i = 1:length(▸(fwdarr))] # FIXME: assumes scalar
  yvals = fwdarr(xvals...)
  opt[:minf](fwdarr, yvals...; callbacks=cbs)
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Dict(:fwdarr => TestArrows.plain_arrows(),
                  :minf => [min_naive, min_domainϵ])
  println(@__FILE__)
  # Makekwrd non standard
  train(optspace,
        @__FILE__;
        toenum=[:fwdarr, :minf],
        runnow=true,
        dorun=initrun,
        nsamples=10,
        group="okletsgo2")
  end

function main()
  genorrun(genopts, initrun)
end

# main()
