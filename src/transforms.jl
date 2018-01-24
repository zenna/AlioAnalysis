plus(x::SubPort) = x
plus(xs::SubPort...) = +(xs...)

"Nnet-expanded parametric inverse of `fwd`"
function netpi(fwd::Arrow, nmabv::NmAbValues = NmAbValues())
  sprtabv = SprtAbValues(⬨(fwd, nm) => abv for (nm, abv) in nmabv)
  invcarr = invert(fwd, inv, sprtabv)
  @grab invcarr
  tabv = traceprop!(invcarr, nmabv)
  @grab tabv
  pslarr = psl(invcarr)
end

"Inverse neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow, tabv::Dict=TraceAbValues())
  UnknownArrow(Symbol(:invnet_, name(fwd)),
                      [name(prt).name for prt in ◂(fwd)],
                      [name(prt).name for prt in ▸(fwd)])
end

"Execution the run"
function stddorun(opt::Dict{Symbol, Any})
  @show fwdarr = opt[:fwdarr]
  invarr = opt[:invarr](fwdarr)
  lossarr = makeloss(invarr, fwdarr, opt[:loss], custϵ=exϵ)
  optimizenet(lossarr, opt[:batch_size])
end
