# function parse_commandline()
#   s = ArgParseSettings()
#   @add_arg_table s begin
#   "--opt1"
#       help = "an option with an argument"
#   "--opt2", "-o"
#       help = "another option with an argument"
#       arg_type = Int
#       default = 0
#   "--flag1"
#       help = "an option without argument, i.e. a flag"
#       action = :store_true
#   "arg1"
#       help = "a positional argument"
#       required = true
#   end
# end

"""
Generate Opts or run opts based on cmdline

`genorun` is typically entry point to a run script.
It has two different modes depending on command line arguments:

1. `julia myrun.jl search`

  Calls `genopts` which eventually calls `dispatchruns`, which will execute `dorun`
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

"""
Run (or schedule to run) `dorun` with diffent optionfiles `opts` ∈ `optspace`

Foreach `opt in optspace`, dispatchruns can
- run `dorun(opt)` locally non blocking in another process (if `runlocal==true`)
- run `dorun(opt)` locally in this process (if `runnow==true`)
- schedule a job on slurm with sbatch (if `runsbatch==true`)
"""
function dispatchruns(optspace,
                      runfile,
                      dorun;
                      toenum=Symbol[],
                      tosample=Symbol[],
                      nsamples=1,
                      runlocal=false,
                      runsbatch=false,
                      runnow=false,
                      group="nogroup",
                      ignoreexceptions=false,
                      runname=()->randrunname(),
                      logdir=()->log_dir(runname=runname, group=group),
                      runpath = joinpath(Pkg.dir("AlioAnalysis", "src", "optim","run.sh")),
                      )
  @showprogress 1 "Computing..." for (i, opt) in enumerate(prodsample(optspace, toenum, tosample, nsamples))
    runname_ = runname()
    logdir_ = logdir()
    optpath = joinpath(logdir_, "$runname_.rd")
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
      if ignoreexceptions
        try
          dorun(opt)
        catch y
          println("Exception caught: $y")
          println("continuing to next run")
        end
      else
        dorun(opt)
      end
    end
  end
end