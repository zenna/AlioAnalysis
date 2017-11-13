using Arrows
using AlioAnalysis
using AlioZoo

# TODO: Batch size is specified in opt, so need to propagate that sumhow all
# the way through to optimize run
# i Dont need a loss function for this example, just domain loss or id loss

function rayrun(opt::Dict{Symbol, Any})
  batch_size = opt.batch_size
  szs = Dict(:sradius => Size([batch_size, 1]),
             :scenter => Size([batch_size, 3]),
             :rdir => Size([batch_size, width * height, 3]),
             :rorig => Size([batch_size, width * height, 3]),
             :doesintersect => Size([batch_size, width * height, 1]),
             :t0 => Size([batch_size, width * height, 1]),
             :t1 => Size([batch_size, width * height, 1]))
   fwdarr = opt[:fwdarr]
   tabv = traceprop!(rsarr, AlioZoo.namesz(rsarr, szs))
   invarr = opt[:invarrgen](fwdarr, tabv)
   lossarr = makeloss(invarr, fwdarr, opt[:loss], custϵ=exϵ)
   optimizerun(lossarr, opt[:batch_size])
end

"Generate data for initialization comparison"
function genopts()
  batch_size = 32
  width = 16
  height = 16
  rsarr = AlioZoo.rayintersect_arr_bcast()
  rsarr, traceprop!(rsarr, AlioZoo.namesz(rsarr, szs))
  optspace = Dict(:fwdarr => rsarr,
                  :batch_size => [16],
                  :invarrgen => [netpi, invnet],
                  :loss => +)
  # Makekwrd non standard
  AlioAnalysis.search(optspace; toenum=[:invarr], runnow=true, dorun=rayrun)
end

function main()
  genorrun(genopts, stdrun)
end

# main()
