using TensorFlowTarget
import TensorFlowTarget: TFTarget

# callbacks
function record_net_weights(data)
  println("Recording net weights")
end

function rec_function(data)
  println("Recording function")
end

function warp(invcarr::CompArrow)
  ϵprt = ◂(invcarr, is(ϵ), 1)
  callbacks = [record_net_weights, rec_function]
  #TODO: add generators
  optimize(invcarr,
           ϵprt,
           [Base.Iterators.repeated(rand(10, 10)) for i = 1:length(▸(invcarr))],
           target=TFTarget;
           callbacks=callbacks)
end

function findnets(carr::CompArrow)
  filter(tarr -> deref(tarr) isa Arrows.UnknownArrow,
         Arrows.simpletracewalk(x->x, carr))
end

function test_warp()
  carr = TestArrows.xy_plus_x_arr()
  invcarr = invert(carr)
  pslarr = psl(invcarr)
  suploss = Arrows.floss(pslarr, (y, x) -> sum(x))
  nnettarr = first(findnets(suploss))
  @show nnettarr
  warp(suploss)
end
