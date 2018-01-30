
using AlioAnalysis
import AlioAnalysis: δpgfx_ny_arr, x_to_y_θ_gen, Sampler, port_sym_names, trainpgfnet, in_port_sym_names
using TensorFlowTarget
using Arrows
import Arrows: pgf
using AlioZoo

batch_size = 2
bitlength = 256
scalarnambv(f::Arrow) = # Assume scalar if no xambv given
  NmAbVals(pnm => AbVals(:size => Size([batch_size, 1, bitlength])) for pnm in port_sym_names(f))

function test_pgf_train(f::Arrow, xabv::XAbVals=scalarnambv(f))
  println("Testing arrow: ", f)
  pgff = Arrows.wraponehot(pgf(f, pgf, xabv), bitlength)
  if isempty(◂(pgff, is(θp)))
    println("Exact inverse, no parameters to predict, skipping!")
    return
  end
  lossarr, n, xabv = δpgfx_ny_arr(f, xabv)
  # xgen = Sampler(()->[rand(get(xabv[nm][:size])...) for nm ∈ in_port_sym_names(f)])
  # xgen = Sampler(()->[rand(0:255, get(xabv[nm][:size])...) for nm ∈ in_port_sym_names(f)])
  xgen = Sampler(()->[rand(0:255, get(xabv[nm][:size])...) for nm ∈ in_port_sym_names(f)])
  @grab xgen = Sampler(()->[Arrows.onehot(rand(0:255, batch_size, 1), bitlength)
                            for nm ∈ in_port_sym_names(f)])

  y_θ_gen = x_to_y_θ_gen(pgff, xgen)
  trainpgfnet(lossarr,
              n,
              y_θ_gen,
              xabv,
              TensorFlowTarget.TFTarget,
              TensorFlowTarget.conv_template;
              cont = data -> data.i < 1000)
end

test_pgf_train(xabv::Tuple{Arrow, XAbVals}) = test_pgf_train(xabv[1], xabv[2])

arrs = [AlioZoo.all_benchmark_arrow_xabv()[1]]

foreach(test_pgf_train, arrs)