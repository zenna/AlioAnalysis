"Preimage attack network to invert `f`"
function pianet(f::Arrow)
  net = UnknownArrow(pfx(f, :pia), out_port_names(f),
                                   in_port_names(f))
end

# PGF predictor
# Find `n: Y -> Θ`
"Arrow that computes pgf loss"
function pgftransform(arr::Arrow, xabv::XAbValues)
  invarr = invert(arr, xabv)
  pgfarr = pgf(arr, xabv)
  net = UnknownArrow(pfx(f, :pgf))
  δny_x_arr(net, :pgflossarr)
end

"Iterator of pgf parameter values from `xgenss`: Iterator over inputs "
function pgfθgen(pgfarr::Arrow, xgens)
  θprts = ◃(pgfarr, is(θp))
  θoids = out_port_id.(θprts)
  pgfjl = il(julia(pgfarr))
  θonly(xs::Vector) = out -> tuple((out[i] for i in θoids)...)  
  imap(θonly ∘ pgfjl, xgens)
end