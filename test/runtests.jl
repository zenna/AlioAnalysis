using AlioAnalysis
using Arrows

import AlioAnalysis: prodsample, netpi, invnet
function test_prodsample()
  optspace = Dict(:fwdarr => [TestArrows.xy_plus_x_arr(), TestArrows.abc_arr()],
                  :batch_size => [16, 32, 64],
                  :invarr => [netpi, invnet],
                  :loss => [+],
                  :check => rand)
  prodsample(optspace, [:batch_size, :invarr], [:check], 1)
end