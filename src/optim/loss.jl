"Construct a loss right inverse which maps inverse domains of `fwd` "
function genloss(invarr::Arrow, fwd::Arrow, loss)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  finv = add_sub_arr!(carr, invarr)
  # idϵ
  foreach(link_to_parent!, ▹(finv))
  finv◃ = ◃(finv, !is(ϵ))
  finv▹ = ▹(finv, !is(θp))
  fwd◃ = fwd(finv◃...)
  # There MUST be a better way
  if fwd◃ isa SubPort
    fwd◃ = [fwd◃]
  end

  # root mean square error, per port
  δ◃s = [mean(δarr()(fwd◃[i], finv▹[i])) for i = 1:length(fwd◃)]
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃s)

  # sum rms over ports
  δtot◃ = plus(δ◃s...)
  add!(ϵ)
  link_to_parent!(δtot◃)
  foreach(link_to_parent!, ◃(finv))
  @assert is_wired_ok(carr)
  return carr

  # Any other loss
  loss◃ = mean(loss(finv◃...))
  (add!(ϵ) ∘ link_to_parent!)(loss◃)

  # Total loss to minimize
  tomin◃ = loss◃ + δtot◃
  link_to_parent!(tomin◃)

  # Link every output output to parent
  foreach(link_to_parent!, ◃(finv))
  @assert is_wired_ok(carr)
  carr
end


"Construct a loss right inverse which maps inverse domains of `fwd` "
function idlo(invarr::Arrow, fwd::Arrow, loss)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  finv = add_sub_arr!(carr, invarr)
  # idϵ
  foreach(link_to_parent!, ▹(finv))
  finv◃ = ◃(finv, !is(ϵ))
  finv▹ = ▹(finv, !is(θp))
  fwd◃ = fwd(finv◃...)
  # There MUST be a better way
  if fwd◃ isa SubPort
    fwd◃ = [fwd◃]
  end

  # root mean square error, per port
  δ◃s = [mean(δarr()(fwd◃[i], finv▹[i])) for i = 1:length(fwd◃)]
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃s)

  # sum rms over ports
  δtot◃ = plus(δ◃s...)
  add!(ϵ)
  link_to_parent!(δtot◃)
  foreach(link_to_parent!, ◃(finv))
  @assert is_wired_ok(carr)
  return carr
end
