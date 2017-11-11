using TensorFlowTarget
import TensorFlowTarget: TFTarget, mlp_template

function warp(carr::CompArrow, batch_size::Integer, template=mlp_template)
  ϵprt = ◂(carr, is(ϵ), 3)
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
           [Base.Iterators.repeated(rand(batch_size, 1)) for i = 1:length(▸(carr))],
           TFTarget;
           cont=data -> data.i < 100,
           callbacks=callbacks)
  [df]
end

"Find `UnknownArrows` within `carr`"
function findnets(carr::CompArrow)
  filter(tarr -> deref(tarr) isa Arrows.UnknownArrow,
         Arrows.simpletracewalk(x->x, carr))
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function makeloss(invarr::Arrow, fwd::Arrow, loss)
  carr = CompArrow(Symbol(:net_loss, name(fwd)))
  net = add_sub_arr!(carr, invarr)
  foreach(link_to_parent!, ▹(net))
  net◃ = ◃(net, !is(ϵ))
  net▹ = ▹(net, !is(θp))
  fwd◃ = fwd(net◃...)
  δ◃ = map(+, fwd◃, net▹)
  loss◃ = loss(net◃...)
  foreach(add!(idϵ) ∘ link_to_parent!, δ◃)
  foreach(add!(ϵ) ∘ link_to_parent!, [loss◃])
  foreach(link_to_parent!, ◃(net, is(ϵ)))
  @assert is_wired_ok(carr)
  carr
end

"nnet-enhanced parametric inverse of `fwd`"
function netpi(fwd::Arrow, loss)
  # Approximately invert fwd and add nnet
  invcarr = aprx_invert(fwd)
  pslarr = psl(invcarr)
  makeloss(pslarr, fwd, loss)
end

function trainnetpi(arr::Arrow = TestArrows.abc_arr(), batch_size=10)
  arr = netpi(arr, +)
  warp(arr, batch_size)
end

"Construct a loss neural network which maps inverse domains of `fwd` "
function invnet(fwd::Arrow, loss)
  unk = UnknownArrow(Symbol(:nnet_, name(fwd)),
                            length(◂(fwd)), length(▸(fwd)))
  makeloss(unk, fwd, loss)
end

"Train a neural network to invert `arr`"
function traininvnet(arr::Arrow = TestArrows.abc_arr(), batch_size=10)
  nnet = invnet(arr, +)
  warp(nnet, batch_size)
end

"Compare a neural network with"
function compare(nruns = 3)
  rundata = [AlioAnalysis.trainanet() for i = 1:nruns]
  rundatajoined = map(AlioAnalysis.joincallbacks, rundata)
  optimal = aa.joinruns(aa.optimal, rundatajoined)
  initial = aa.joinruns(aa.init, rundatajoined)
  allnetdata = manyappend!(rundatajoined)
  allnetloss = allnetdata[:loss]

  # Noww with PI

  # • Get data from PI too
  # • Have comparable loss term
  # • Record suploss, idloss, domainloss
  # • Optimize wrt
  # • Match number of parameters
  # • Automate on openmind
end
