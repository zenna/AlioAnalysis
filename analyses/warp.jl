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
