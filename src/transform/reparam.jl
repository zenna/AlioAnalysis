# Reparamterization of a neural network

# FIXME: Move these elsewhere
port_names(arr) = [nm.name for nm in name.(ports(arr))]
in_port_names(arr) = [nm.name for nm in name.(in_ports(arr))]
out_port_names(arr) = [nm.name for nm in name.(out_ports(arr))]

"Network from `Y -> \Theta`"
function pslnet(invf::Arrow, xabv::XAbValues)
  net = UnknownArrow(pfx(invf, :psl), in_port_names(invf),
                                      in_port_names(f))
end

"Parameter Selecting Function: `psl: Y -> θ` from `invf: Y x θ -> X`"
pslnet(invf::Arrow) = UnknownArrow(pfx(invf, :psl), ▸(invf, !is(θp)), ▸(invf, is(θp)))

# FIXME: Singleton Problem
linkmany(x::Vector{SubPort}, y::Vector{SubPort}) = foreach(⥅, x, y)
linkmany(x::SubPort, y::Vector{SubPort}) = (@pre issingleton(y); x ⥅ y[1])

"Compose `psl`` with `invf` to yield reparamterized`"
function reparamf(psl::Arrow, invf::Arrow)
  @pre Set(port_sym_name.(⬧(psl))) == Set(port_sym_name.(▸(invf)))
  invfrepram = CompArrow(:reparam, ▸(psl), ◂(invf)) # Y -> X
  pslθ◃s = psl(▹(invfrepram)...)
  # Connect ports by nazme
  to_invf = [pslθ◃s; ▹(invfrepram)] # Θ x Y: To link to inprts of invf
  nm_to_invf = Dict{Symbol, SubPort}(port_sym_name(sprt) => sprt for sprt in to_invf)
  xprts = invf(nm_to_invf)
  linkmany(xprts, ◃(invfrepram))
  @grab invfrepram
  @post invfrepram is_valid(invfrepram)
end

"Invert `arr`, construct `psl` and `reparamterizes`"
function reparamloss(arr::Arrow, xabv::XAbValues) # F -> reparameterized inverse 
  # Invert and compose with psl (parameter selecting function)
  invf = invert(arr, inv, xabv)
  psl = pslnet(invf)
  pianetarr = reparamf(psl, invf)

  # Create the loss arrow
  lossarr = nlossarr(arr, pianetarr)

  # Find invf in the loss arrow and propagate with its abvalues to get abvalues for net 
  invfinlossar = first(Arrows.findtarrs(lossarr, invf))
  tabv = Arrows.tabvfromxabv(invfinlossar, xabv)
  lossarr, tabv
end