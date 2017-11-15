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

dorun(optpath::String) = dorun(loadopt(optpath))

"Log directory, e.g. ~/datadir/mnist/Oct14_02-43-22_my_comp/"
function log_dir(;root=datadir(), jobid=randstring(5), group="nogroup", comment="")
  logdir = join([jobid,
                 now(),
                 gethostname(),
                 comment],
                 "_")
  joinpath(root, "runs", group, logdir)
end

function train(optspace;
                toenum=Symbol[],
                tosample=Symbol[],
                nsamples=1,
                runlocal=false,
                runsbatch=false,
                runnow=false,
                dorun=stddorun,
                group="nogroup")
  for (i, opt) in enumerate(prodsample(optspace, toenum, tosample, nsamples))
    jobid = randstring(5)
    logdir = log_dir(jobid=jobid, group=group)
    optpath = joinpath(logdir, "options.opt")
    runpath = "/home/zenna/repos/Alio/AlioAnalysis.jl/src/run.sh"
    thisfile = "/home/zenna/repos/Alio/AlioAnalysis.jl/runs/init.jl"
    mkpath(logdir)
    saveopt(optpath, opt)
    println("Saving options at: ", optpath)
    if runsbatch
      cmd =`sbatch -J $jobid -o $jobid.out $runpath $thisfile $optpath`
      println("Scheduling sbatch: ", cmd)
      run(cmd)
    end
    if runlocal
      cmd = `julia $thisfile $optpath`
      println("Running: ", cmd)
      run(cmd)
    end
    if runnow
      @show opt
      dorun(opt)
    end
  end
end
