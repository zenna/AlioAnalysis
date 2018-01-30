## Misc Utils

Options = Dict{Symbol, Any} # FIXME: reprecate in place of rundata

"Enumerate product space of `toenum` sampling from `tosampale`"
function prodsample(optspace,
                    toenum::Vector{Symbol},
                    tosample::Vector{Symbol},
                    nsamples::Integer = 1)
  toenumprod = filter((k, v)->k ∈ toenum, optspace)
  tosample = filter((k, v)-> k ∈ tosample, optspace)
  iter = Iterators.product(values(toenumprod)...)
  dicts = []
  for it in iter
    subdict1 = Dict(zip(keys(toenumprod), it))
    for i = 1:nsamples
      subdict2 = Dict{Symbol, Any}(k => optspace[k]() for k in keys(tosample))
      subdict = merge(optspace, subdict1, subdict2)
      push!(dicts, subdict)
    end
  end
  dicts
end

"Turn a key value into command line argument"
function stringify(k, v)
  if v == true
    "--$k"
  elseif v == false
    ""
  else
    "--$k=$v"
  end
end

function linearstring(d::Dict, ks::Symbol...)
  join([string(k, "_", d[k]) for k in ks], "_")
end

"Data Directory. Defaults to `homedir()` if DATADIR not an environment variable"
function datadir()
  if "DATADIR" in keys(ENV)
    ENV["DATADIR"]
  else
    homedir()
  end
end

pathfromgroup(group;root=datadir()) = joinpath(root, "runs", group)

randrunname(len=5)::Symbol = Symbol(:run_, randstring(len))

"Log directory, e.g. ~/datadir/mnist/Oct14_02-43-22_my_comp/"
function log_dir(;root=datadir(), runname=randrunname(), group="nogroup", comment="")
  logdir = join([runname,
                 now(),
                 gethostname(),
                 comment],
                 "_")
  joinpath(root, "runs", group, logdir)
end

"""
Constant function from anything to `y`

```jldoctest
julia> g = constfunction(4)
(::#4) (generic function with 1 method)

julia> g(3)
4
```
"""
constfunction(y) = (x...) -> y

# Arrows related utils
"Find `UnknownArrows` within `carr`"
function findnets(carr::CompArrow)
  filter(tarr -> deref(tarr) isa Arrows.UnknownArrow,
         Arrows.simpletracewalk(x->x, carr))
end
