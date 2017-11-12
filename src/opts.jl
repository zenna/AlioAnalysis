# Standard Opts

# Standard Optiosn
function stdopts()
  fwdarr = TestArrows.plain_arrows()
  trainsize = [1, 2, 3]
  @NT(fwdarr = fwdarr, trainsize = trainsize)
end

"PGF specific parameter"
function piopts()
  pgfinit = [true, false]   # Use PGF to initialize search
  tomin =   [:domainloss, :idloss]
end

function nnetopts()
end