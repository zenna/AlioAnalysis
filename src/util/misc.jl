using ProgressMeter

Options = Dict{Symbol, Any}
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

"""
Generate Opts or run opts based on cmdline

`genorun` is typically entry point to a run script.
It has two different modes depending on command line arguments:

1. `julia myrun.jl search`

  Calls `genopts` which eventually calls `train`, which will execute `dorun`
  on diffent options

2. `julia myrun /path/to/options.opt`

  Will call dorun(opt) where opt is loaded from file

Together these allow both running scripts on local machine and scheduling them.
Scheduling involves saving an options file to disk and calling sbatch with the
same script but the options file as the command line argument
"""
function genorrun(genopts, dorun)
  if length(ARGS) != 1
    println("Wrong num arguments, should be 1 but was $(length(ARGS))")
  elseif ARGS[1] == "search"
    genopts()
  else
    opt = loaddict(ARGS[1])
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

dorun(optpath::String) = dorun(loaddict(optpath))

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
Run (or schedule to run) `dorun` with diffent optionfiles `opts` ∈ `optspace`

Foreach `opt in optspace`, train can
- run `dorun(opt)` locally non blocking in another process (if `runlocal==true`)
- run `dorun(opt)` locally in this process (if `runnow==true`)
- schedule a job on slurm with sbatch (if `runsbatch==true`)
"""
function train(optspace, #FIXME: take in input, SRL and rename to more meaningful
               runfile;
               toenum=Symbol[],
               tosample=Symbol[],
               nsamples=1,
               runlocal=false,
               runsbatch=false,
               runnow=false,
               dorun=stddorun,
               group="nogroup",
               ignoreexceptions=false,
               runname=()->randrunname(),
               logdir=()->log_dir(runname=runname, group=group))
  @showprogress 1 "Computing..." for (i, opt) in enumerate(prodsample(optspace, toenum, tosample, nsamples))
    runname_ = runname()
    logdir_ = logdir()
    optpath = joinpath(logdir_, "options.opt")
    runpath = joinpath(Pkg.dir("AlioAnalysis", "src", "optim","run.sh"))
    mkpath(logdir_)
    opt[:group] = group
    opt[:runname] = runname_
    opt[:logdir] = logdir_
    opt[:file] = runfile
    savedict(optpath, opt)
    println("Saving options at: ", optpath)
    if runsbatch
      cmd =`sbatch -J $runname_ -o $runname_.out $runpath $runfile $optpath`
      println("Scheduling sbatch: ", cmd)
      run(cmd)
    end
    if runlocal
      cmd = `julia $runfile $optpath`
      println("Running: ", cmd)
      run(cmd)
    end
    if runnow
      @show opt
      dorun(opt)
      # try
      # catch y
      #   println("Exception caught: $y")
      #   if !ignoreexceptions
      #     throw(y)
      #   end
      #   println("continuing to next run")
      # end
    end
  end
end

## Misc Utils
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
