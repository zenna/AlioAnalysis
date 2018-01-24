using Arrows
using Arrows.TestArrows
import AlioAnalysis: id_loss, floss
using Base.Test

function pre_test(arr::Arrow)
  println("Testing arrow ", name(arr))
  arr
end

function test_exact_inverse()
  fwdarr = TestArrows.xy_plus_x_arr()
  invarr = TestArrows.inv_xy_plus_x_arr()
  lossarr = id_loss(fwdarr, invarr)
  @test lossarr(1.0, 2.0) == 0
end

test_exact_inverse()

function test_floss(arr::Arrow)
  invarr = aprx_invert(arr) # dont totalize
  sumxs(ys, xs) = sum(xs)
  floss(invarr, sumxs)
end

foreach(test_floss ∘ pre_test, plain_arrows())

function test_id_loss()
  sin_arr = Arrows.TestArrows.sin_arr()
  aprx = Arrows.aprx_invert(sin_arr)
  lossarr = id_loss(sin_arr, aprx)
  @test is_valid(lossarr)
end

test_id_loss()
