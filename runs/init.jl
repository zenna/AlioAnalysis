# TODO:
# record domain loss and id loss
# record domain loss of each primitive
# vary contractor function
#

using Arrows
using AlioZoo
using AlioAnalysis
import AlioAnalysis: min_naive, recordrungen, min_domainϵ, everyn, savedfgen,
                     printloss, Options, min_superϵ, min_domϵ, min_idϵ, min_both

function benchmarkarrs()
  [TestArrows.xy_plus_x_arr(),
  # TestArrows.twoxy_plus_x_arr(),
  TestArrows.abc_arr(),
  TestArrows.weird_arr()]
end

function initrun(opt::Dict{Symbol, Any})
  println("Starting init")
  df, record = recordrungen()
  cbs = [record,
         everyn(savedfgen("rundata", joinpath(opt[:logdir], "rundata.jld2"), df), 3),
         everyn(printloss, 5)]
  fwdarr = opt[:fwdarr]
  @grab fwdarr
  println("MIN!!!!!!!F", opt[:minf])
  pgfarr = Arrows.pgf(fwdarr)
  pgfxvals = [rand() for i = 1:length(▸(fwdarr))] # FIXME: assumes scalar
  xvals = [rand() for i = 1:length(▸(fwdarr))] # FIXME: assumes scalar
  @show pgfvals = pgfarr(pgfxvals...)
  yvals = fwdarr(xvals...)
  opt[:minf](fwdarr, yvals...; callbacks=cbs, opt=opt, pgfvals=pgfvals)
  println("Finishing init")
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Options(:fwdarr => AlioZoo.stanford_arr(),
                    #  :minf => [])
                     :minf => [min_idϵ, min_domϵ, min_naive, min_both])
  println(@__FILE__)
  # Makekwrd non standard
  dispatchruns(optspace,
        @__FILE__;
        toenum=[:minf],
        runnow=true,
        runlocal=false,
        dorun=initrun,
        nsamples=1,
        group="stanfordpray",
        ignoreexceptions=false)
  end

function main()
  genorrun(genopts, initrun)
end

# genopts()
# main()
