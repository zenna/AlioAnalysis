using TensorFlowTarget
import TensorFlowTarget: TFTarget, mlp_template

function warp(carr::CompArrow, batch_size::Integer, template=mlp_template)
  ϵprt = ◂(carr, is(exϵ), 1)
  # Find the network and add `func`
  nnettarr = first(findnets(carr))
  insizes = [Size([batch_size, 1]) for i = 1:length(▸(deref(nnettarr)))]
  outsizes = [Size([batch_size, 1]) for i = 1:length(◂(deref(nnettarr)))]
  deref(nnettarr).func = args->mlp_template(args, insizes, outsizes)

  # Setup callbacks
  df, std_cb = savedata()
  callbacks = [std_cb]

  # Optimize
  optimize(carr,
           ϵprt,
           [Arrows.Sampler{Array}(()->rand(batch_size, 1)) for i = 1:length(▸(carr))],
           TFTarget;
           cont=data -> data.i < 400,
           callbacks=callbacks)
  [df]
end

"Find `UnknownArrows` within `carr`"
function findnets(carr::CompArrow)
  filter(tarr -> deref(tarr) isa Arrows.UnknownArrow,
         Arrows.simpletracewalk(x->x, carr))
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function makeloss(invarr::Arrow, fwd::Arrow, loss; custϵ=ϵ)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  net = add_sub_arr!(carr, invarr)
  foreach(link_to_parent!, ▹(net))
  net◃ = ◃(net, !is(ϵ))
  net▹ = ▹(net, !is(θp))
  fwd◃ = fwd(net◃...)
  δ◃ = map(+, fwd◃, net▹)
  loss◃ = loss(net◃...)
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃)
  foreach(add!(custϵ) ∘ link_to_parent!, [loss◃])
  foreach(link_to_parent!, ◃(net, is(ϵ)))
  @assert is_wired_ok(carr)
  carr
end

"Example Error"
struct ExError <: Arrows.Err end
exϵ = ExError
Arrows.superscript(::Type{ExError}) = :ᵉˣᵋ

"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow, loss)
  # Approximately invert fwd and add nnet
  invcarr = aprx_invert(fwd)
  pslarr = psl(invcarr)
  makeloss(pslarr, fwd, loss; custϵ=exϵ)
end

function trainnetpi(arr::Arrow = TestArrows.abc_arr(), batch_size=100)
  arr = netpi(arr, +)
  warp(arr, batch_size)
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow, loss)
  unk = UnknownArrow(Symbol(:nnet_, name(fwd)),
                            length(◂(fwd)), length(▸(fwd)))
  makeloss(unk, fwd, loss, custϵ=exϵ)
end

"Train a neural network to invert `arr`"
function traininvnet(arr::Arrow = TestArrows.abc_arr(), batch_size=100)
  nnet = invnet(arr, +)
  warp(nnet, batch_size)
end

"Compare a neural network with"
function alldata(rundata)
  rundatajoined = map(joincallbacks, rundata)
  optimal = joinruns(AlioAnalysis.optimal, rundatajoined)
  initial = joinruns(init, rundatajoined)
  allnetdata = manyappend(rundatajoined...)
  allnetloss = allnetdata[:loss]
end

"Do TSNE on multiple runs"
function tsneruns(losses...)
  @show alllosses = vcat(losses...)
  points = TSne.tsne(Array(alllosses), (x, y)->(sum(abs.(x-y))))
end

## Example

function compare(nruns = 3)
  invnetrundata = [traininvnet() for i = 1:nruns]
  netpirundata = [trainnetpi() for i = 1:nruns]
  losses = map(alldata, (invnetrundata, netpirundata))
  points = tsneruns(losses...)
  scattermany(points[1:length(losses[1]), :], map(joincallbacks, invnetrundata), true, :circle)
  scattermany(points[length(losses[1]):end, :], map(joincallbacks, netpirundata), false, :xcross)
  # invnetrundata, netpirundata, points, losses
end

  # Noww with PI

  # • Get data from PI too
  # • Have comparable loss term
  # • Record suploss, idloss, domainloss
  # • Optimize wrt
  # • Match number of parameters
  # • Automate on openmind
