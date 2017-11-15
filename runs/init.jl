using Arrows
using AlioAnalysis: plus, optimizerun, genloss
using AlioZoo
using AlioAnalysis

function rayrun(opt::Dict{Symbol, Any})
  szs = Dict(:sradius => Size([batch_size, 1, 1]),
             :scenter => Size([batch_size, 1, 3]),
             :rdir => Size([batch_size, width * height, 3]),
             :rorig => Size([batch_size, width * height, 3]),
             :doesintersect => Size([batch_size, width * height, 1]),
             :t0 => Size([batch_size, width * height, 1]),
             :t1 => Size([batch_size, width * height, 1]))
  nmabv = NmAbValues(nm => AbValues(:size => val) for (nm, val) in szs)
  fwdarr = opt[:fwdarr]
  invarr = opt[:invarrgen](fwdarr, nmabv)
  lalaloss(⬨s...) = abs(plus(⬨s...)) # minimize the norm
  lossarr = genloss(invarr, fwdarr, lalaloss)
  tabv = traceprop!(lossarr, nmabv)
  optimizerun(lossarr, xabv=nmabv)
end

"Generate data for initialization comparison"
function genopts()
  # Vary over different arrows, varying the initial conditions
  optspace = Dict(:fwdarr => TestArrows.plain_arrows(),
                  :invarrgen => [netpi, invnet],
                  :loss => +)
  # Makekwrd non standard
  train(optspace;
         toenum=[:invarrgen],
         runnow=true,
         dorun=rayrun,
         nsamples=2,
         group="raytrace")
  end

function main()
  genorrun(genopts, stdrun)
end

# main()
