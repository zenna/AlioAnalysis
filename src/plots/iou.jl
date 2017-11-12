# Intersection over union for voxel renderer
using LaTeXStrings

function fake_iou()
  df = DataFrame(Method = ["Parametric Inversion", "Soltani Method"],
                 IoU = [rand(), rand()])
  Base.reprmime(MIME("text/latex"), df)
end

function fake_testtraintable()
  d = 1:10
  df = DataFrame(D = d,
                 pi_pgf_dom = rand(length(d)),
                 pi_pgf_id =  rand(length(d)),
                 net = rand(length(d)))
  Base.reprmime(MIME("text/latex"), df)
end

"Fake table showing runtime performance"
function fake_testtraintable()
  d = 1:10
  df = DataFrame(D = d,
                 pi_pgf_dom = rand(length(d)),
                 pi_pgf_id =  rand(length(d)),
                 net = rand(length(d)))
  Base.reprmime(MIME("text/latex"), df)
end
