# DataFrame Utils

"Recursive join on `on` of many dataframes `dfs`"
function manyjoin(on::Symbol, dfs::DataFrame...)
  x = first(dfs)
  for df in dfs[2:end]
    x = join(x, df, on=on)
  end
  x
end

"Append one or more dataframes to `df1`"
function manyappend(df1::DataFrame, df2::DataFrame, dfs::DataFrame...)
  df1 = deepcopy(df1)
  dfs = vcat(df2, [dfs...])
  foreach(df -> append!(df1, df), dfs)
  df1
end

"Find `UnknownArrows` within `carr`"
function findnets(carr::CompArrow)
  filter(tarr -> deref(tarr) isa Arrows.UnknownArrow,
         Arrows.simpletracewalk(x->x, carr))
end

"Example Error"
struct ExError <: Arrows.Err end
exϵ = ExError
Arrows.superscript(::Type{ExError}) = :ᵉˣᵋ

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
      subdict2 = Dict(k => optspace[k]() for k in keys(tosample))
      subdict = merge(optspace, subdict1, subdict2)
      push!(dicts, subdict)
    end
  end
  dicts
end

"Save opt to file"
function saveopt(poth, opt)
  open(poth, "w") do f
    serialize(f, opt)
  end
end

"Load optiosn from a file"
function loadopt(path)
  deserialize(open(path))
end

"Generate Opts or run opts based on cmdline"
function genorrun(genopts, dorun)
  if length(ARGS) != 1
    println("Wrong num arguments, should be 1 but was $(length(ARGS))")
  elseif ARGS[1] == "search"
    genopts()
  else
    opt = loadopt(ARGS[1])
    dorun(opt)
  end
end

"Data Dir"
function datadir()
  if "DATADIR" in keys(ENV)
    ENV["DATADIR"]
  else
    homedir()
  end
end

"Log directory, e.g. ~/datadir/mnist/Oct14_02-43-22_my_comp/"
function log_dir(;root=datadir(), jobid=randstring(5), group="nogroup", comment="")
  logdir = join([randstring(5),
                 now(),
                 gethostname(),
                 comment],
                 "_")
  joinpath(root, "runs", group, logdir)
end
