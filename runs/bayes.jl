# Test hypothesis that pi imposes a structural prior on function space

using Arrows
using AlioAnalysis
using AlioZoo
import AlioAnalysis: plus, optimizerun

# TODO: Batch size is specified in opt, so need to propagate that sumhow all
# the way through to optimize run
# i Dont need a loss function for this example, just domain loss or id loss

function bayesrun(opt::Dict{Symbol, Any})
   fwdarr = opt[:fwdarr]
   portnms = name.(⬧(fwdarr))
   # Scalar examples
   nmabv = NmAbValues(nm.name => AbValues(:size => Size([opt[:batch_size], 1])) for nm in portnms)
   invarr = opt[:invarrgen](fwdarr, nmabv)
   lalaloss(⬨s...) = abs(plus(⬨s...)) # minimize the norm
   lossarr = AlioAnalysis.genloss(invarr, fwdarr,  lalaloss)
   @show lossarr
   optimizerun(lossarr,
               xabv=nmabv)
end

"Generate data for initialization comparison"
function genopts()
  optspace = Dict(:fwdarr => TestArrows.plain_arrows()[1:3],
                  :trainsz => [1, 2, 8, 32, 128, 512],
                  :batch_size => 32,
                  :invarrgen => [netpi, invnet])
  # Makekwrd non standard
  train(optspace;
         toenum=[:trainsz, :fwdarr, :invarrgen],
         runnow=true,
         dorun=bayesrun,
         nsamples=2,
         group="bayes")
end

function main()
  genorrun(genopts, stdrun)
end

# main()
