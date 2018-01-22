using AlioAnalysis
using Arrows
using Spec
import AlioAnalysis: min_naive, recordrungen, min_domainϵ, everyn, savedfgen,
                     printloss, Options, min_superϵ, min_domϵ, min_idϵ, min_both

function test_analysis()
  with_pre() do
    arrs = [TestArrows.xy_plus_x_arr(), TestArrows.twoxy_plus_x_arr()]

    function initrun(opt::Dict{Symbol, Any})
      df, record = recordrungen(opt[:runname])
      cbs = [record,
             everyn(savedfgen("rundata", joinpath(opt[:logdir], "rundata.jld2"), df), 3),
             everyn(printloss, 5)]
      fwdarr = opt[:fwdarr]

      # Generate input (to forward arrow) data
      xvals = [rand() for i = 1:length(▸(fwdarr))] # FIXME: assumes scalar

      # Generate output data (input to inverse)
      yvals = fwdarr(xvals...)

      min_naive(fwdarr, yvals...; callbacks=cbs, opt=opt)
      println("Finishing init")
    end

    "Generate data for initialization comparison"
    function genopts()
      # Vary over different arrows, varying the initial conditions
      optspace = Options(:fwdarr => arrs,
                        #  :minf => [])
                         :minf => [min_idϵ, min_domϵ, min_naive, min_both])
      println(@__FILE__)
      # Makekwrd non standard
      train(optspace,
            @__FILE__;
            toenum=[:fwdarr],
            runsbatch=false,
            runnow=true,
            runlocal=false,
            dorun=initrun,
            nsamples=1,
            group="test_analysis",
            ignoreexceptions=false,
            logdir=()->"/tmp/aliotest")
    end
    genopts()
  end
end

# test_analysis()
