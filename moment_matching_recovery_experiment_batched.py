'''
moment_matching_recovery_experiment_batched.py

End-to-end execution of a batch of moment matching curve recovery experiments.
'''

import click
import os
import json
import pandas as pd
from optim import *
from jax import config
config.update("jax_enable_x64", True)
import time


def run_single_experiment(seed, d, M, sigma2, use_true_moments, N, chunk_size, length_gd_nit, finetune_nit, num_baseline_trials):
  '''
  Run an experiment recovering a single curve. Basically the same as main() in 
    third_moment_single_recovery_experiment.py without the logging. Maybe this is bad software, 
    but I'm not a software engineer.
  '''
  t0 = time.time()
  
  # initialize RNG
  key, subkey = random.split(random.PRNGKey(seed))

  # generate the curve
  seg_lens = random.uniform(subkey, shape = (M,), minval = 1, maxval = 2)
  key, subkey = random.split(key)
  C = gen_curve_sphere_sampling(seed, M, d, seg_lens)
  C = C - compute_m1(C)

  # compute the moments of the curve
  if use_true_moments:
    # use moments of zero-noise curves
    m1 = compute_m1(C)
    m2 = compute_m2(C)
    m3 = compute_m3(C)
  else:
    m1_raw, m2_raw, m3_raw, _ = gen_cloud_moments(subkey, C, sigma2, N, num_chunks = N // chunk_size, N_mini = 0)
    m1 = m1_raw
    m2 = m2_raw - sigma2 * jnp.eye(d) # de-bias second and third moments
    m3 = m3_raw - 3 * sigma2 * compute_sym_part(jnp.einsum("i, jk -> ijk", m1, jnp.eye(d)))  

  # estimate the subspaces
  subspaces, status = recover_C_subspaces(m3, m2, M, seed = 0)

  # estimate the curve from subspaces alone
  Lhat_init = jnp.ones(shape = (M+1,))
  Chat_subspaces = estimate_C_from_subspaces(subspaces, Lhat_init, m1, m2, m3, nit = length_gd_nit)

  curve_loss_subspace_estimation = compute_curve_loss(C, Chat_subspaces)
  m3_loss_subspace_estimation = m3_loss(Chat_subspaces, compute_m3(C))
  
  # project the predicted curve and moments into M-dimensional subspace
  basis = get_basis(m2, M)
  Chat_projected = project_to_subspace(Chat_subspaces.T, basis).T
  if use_true_moments:
    C_projected = project_to_subspace(C.T, basis).T
    m1_projected = compute_m1(C_projected)
    m2_projected = compute_m2(C_projected)
    m3_projected = compute_m3(C_projected)
  else:
    m1_projected, m2_projected, m3_projected = project_moments(m1, m2, m3, M)

  # perform optimization on third moment directly on curve points within the subspace
  phat = compute_seg_lens(Chat_projected) # phat is same for projected vs unprojected curve
  phat = phat / jnp.sum(phat)
  Chat_projected, phat = finetune_C_with_moments(Chat_projected, phat, m3_projected, nit = finetune_nit)
  Chat = deproject_from_subspace(Chat_projected.T, basis).T
  curve_loss_m3_finetuning = compute_curve_loss(C, Chat)
  m3_loss_m3_finetuning = m3_loss(Chat, compute_m3(C))
  
  # compute two baselines by trialing a large number of randomly initialized Chats in the
  # lower dimensional subspace and optimizing either third moment alone or all three moments and
  # choosing the best result
  best_Chat_m123, best_Chat_m3 = estimate_C_baseline(seed, m1_projected, m2_projected, m3_projected, M, num_trials = num_baseline_trials)
  Chat_baseline_m123 = deproject_from_subspace(best_Chat_m123.T, basis).T
  Chat_baseline_m3 = deproject_from_subspace(best_Chat_m3.T, basis).T

  curve_loss_m123_baseline = compute_curve_loss(C, Chat_baseline_m123)
  m3_loss_m123_baseline = m3_loss(Chat_baseline_m123, compute_m3(C))
  curve_loss_m3_baseline = compute_curve_loss(C, Chat_baseline_m3)
  m3_loss_m3_baseline = m3_loss(Chat_baseline_m3, compute_m3(C))
  
  t1 = time.time()

  results_dict = {
    "seed" : seed,
    "status" : status,
    "curve_loss_subspace_estimation" : curve_loss_subspace_estimation.item(),
    "m3_loss_subspace_estimation" : m3_loss_subspace_estimation.item(),
    "curve_loss_m3_finetuning" : curve_loss_m3_finetuning.item(),
    "m3_loss_m3_finetuning" : m3_loss_m3_finetuning.item(),
    "curve_loss_m123_baseline" : curve_loss_m123_baseline.item(),
    "m3_loss_m123_baseline" : m3_loss_m123_baseline.item(),
    "curve_loss_m3_baseline" : curve_loss_m3_baseline.item(),
    "m3_loss_m3_baseline" : m3_loss_m3_baseline.item(),
    "runtime": t1 - t0
  }

  return results_dict


@click.command(context_settings={'show_default': True})
@click.option("--exp_name", type = str, default = "default_experiment", help = "name of experiment")
@click.option("--num_experiments", type = int, default = 500, help = "number of experiments to run")
@click.option("--seed", type = int, default = 88, help = "seed for RNG")
@click.option("--d", type = int, default = 64, help = "ambient dimension of curve")
@click.option("--M", type = int, default = 32, help = "number of segments in curve")
@click.option("--sigma2", type = float, default = 0.25, help = "noise level")
@click.option("--use_true_moments", type = bool, default = False, help = "cheat by using ground truth curve moments rather than moments estimated from a cloud")
@click.option("--N", type = int, default = 100_000_000, help = "number of points in point cloud around curve to generate")
@click.option("--chunk_size", type = int, default = 5_000_000, help = "chunk size for chunked cloud moment computation")
@click.option("--length_gd_nit", type = int, default = 100, help = "number of gradient descent steps to take in estimate_C_from_subspaces()")
@click.option("--finetune_nit", type = int, default = 2500, help = "number of gradient descent steps to take in finetune_C_with_moments()")
@click.option("--num_baseline_trials", type = int, default = 10, help = "number of random trials to run for baseline predictions")
def main(exp_name, num_experiments, seed, d, m, sigma2, use_true_moments, n, chunk_size, length_gd_nit, finetune_nit, num_baseline_trials):
  args = locals()

  if not os.path.exists("outputs/"):
    os.mkdir("outputs")

  if not os.path.exists("outputs/outputs_batch_recovery/"):
    os.mkdir("outputs/outputs_batch_recovery")

  if m >= d:
    raise Exception(f"number of segments M must be less than ambient dimension d, got M = {m}, d = {d}")
  
  # make subdirectory to save results in
  outdir = f"outputs/outputs_batch_recovery/{exp_name}"
  if not os.path.exists(outdir):
    os.mkdir(f"outputs/outputs_batch_recovery/{exp_name}")
  print(f"saving results to {outdir}", flush = True)

  # save arguments for posterity
  args["N"] = args.pop("n") # click annoyingly converts all args to lowercase
  args["M"] = args.pop("m")
  with open(os.path.join(outdir, "args.json"), "w") as f:
    json.dump(args, f)
  M = m
  N = n

  # generate seeds for individual experiments
  seeds = random.choice(random.PRNGKey(seed), jnp.arange(seed * int(1e6), (seed + 1) * int(1e6)), shape = (num_experiments,), replace = False)

  df = pd.DataFrame()

  # run an experiment and add results to dataframe
  T0 = time.time()
  for seed in tqdm(seeds, desc = "running experiments..."):
    results_dict = run_single_experiment(seed, d, M, sigma2, use_true_moments, N, chunk_size, length_gd_nit, finetune_nit, num_baseline_trials)
    df = pd.concat([df, pd.DataFrame([results_dict])])
  T1 = time.time()
  print(f"Total runtime: {(T1-T0):2f} seconds", flush = True)

  # clean up dataframe, add means to dict, count numer of occurences of each status, save to json
  df = df.set_index(["seed"])

  mean = df.mean(axis=0).to_dict()
  mean.pop("status")
  results = {}
  for key, value in mean.items():
    results[f"MEAN_{key}"] = value

  try:
    num0 = df["status"].value_counts()[0].item()
  except:
    num0 = 0
  try:
    num1 = df["status"].value_counts()[1].item()
  except:
    num1 = 0
  try:
    num2 = df["status"].value_counts()[2].item()
  except:
    num2 = 0

  results["num_experiments"] = len(df)
  results["num_status_0"] = num0
  results["num_status_1"] = num1
  results["num_status_2"] = num2
  results["total_exp_time_s"] = T1 - T0

  df.to_csv(os.path.join(outdir, "df.csv"))
  losses_df = df[["curve_loss_subspace_estimation", "curve_loss_m3_finetuning", "curve_loss_m3_baseline", "curve_loss_m123_baseline", "m3_loss_subspace_estimation", "m3_loss_m3_finetuning", "m3_loss_m3_baseline", "m3_loss_m123_baseline"]]
  summary_df = losses_df.describe().loc[['25%', '50%', '75%']].T
  summary_df.to_csv(os.path.join(outdir, "summary_df.csv"))

  with open(os.path.join(outdir, "results.json"), "w") as f:
    json.dump(results, f)




if __name__ == "__main__":
  t0 = time.time()
  main()
  t1 = time.time()
  print(f"full experiment took {(t1 - t0):.2f} seconds")


  



