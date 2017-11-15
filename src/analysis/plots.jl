"Scatter many points"
function scattermany(points, results, newplot::Bool, markershape=:circle)
  @show size(points)
  @show size.(results)
  lb = 1
  local plot
  for res in results
    scat = lb == 1 && newplot ? scatter : scatter!
    ub = lb+size(res, 1)-1
    @show lb, ub
    pts = points[lb:ub, :]
    plot = scat(pts[:,1], pts[:,2], markershape=markershape)
    lb = ub + 1
  end
  plot
end
