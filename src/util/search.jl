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

"Turn options into a string that can be passed on command line"
function make_batch_string(options)
  batch_string = [stringify(k, v) for (k, v) in options]
end

"Execute sbatch with options"
function run_sbatch(file_path,
                    options,
                    sbatch_opt = Dict(),
                    bash_run_path = join(3, "run.sh"))
  run_str = vcat("sbatch", make_batch_string(sbatch_opt), bash_run_path, file_path, make_batch_string(options))
  print(run_str)
  subprocess.call(run_str)
end

"Execute process with options"
function run_local_batch(file_path, options, blocking=true)
  run_str = ["julia", file_path] + make_batch_string(options)
  print("Subprocess call:", run_str)
  if blocking
    subprocess.call(run_str)
  else
    subprocess.Popen(run_str)
  end
end