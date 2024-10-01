'''
moment_matching_recovery_experiment.py

End-to-end execution of third moment-based recovery experiment for a single curve.
'''

import click
import os
import json
from optim import *
from jax import config
config.update("jax_enable_x64", True)
import time
import numpy as np
import matplotlib.pyplot as plt

@click.command(context_settings={'show_default': True})
@click.option("--exp_name", type = str, default = "default_experiment", help = "name of experiment")
@click.option("--seed", type = int, default = 88, help = "seed for RNG")
@click.option("--d", type = int, default = 12, help = "ambient dimension of curve")
@click.option("--M", type = int, default = 8, help = "number of segments in curve")
@click.option("--sigma2", type = float, default = 0.25, help = "noise level")
@click.option("--use_true_moments", type = bool, default = True, help = "cheat by using ground truth curve moments rather than moments estimated from a cloud")
@click.option("--N", type = int, default = 10_000_000, help = "size of point cloud to generate")
@click.option("--N_mini", type = int, default = 1_000, help = "number of points in mini point cloud to plot")
@click.option("--chunk_size", type = int, default = 100_000, help = "chunk size for chunked cloud moment computation")
@click.option("--length_gd_nit", type = int, default = 100, help = "number of gradient descent steps to take in estimate_C_from_subspaces()")
@click.option("--finetune_nit", type = int, default = 2500, help = "number of gradient descent steps to take in finetuning step")
@click.option("--num_baseline_trials", type = int, default = 10, help = "number of random trials to run for baseline predictions")
@click.option("--generate_plots", type = bool, default = True, help = "whether or not to generate a basic maplotlib plot of the results")
def main(exp_name, seed, d, m, sigma2, use_true_moments, n, n_mini, chunk_size, length_gd_nit, finetune_nit, num_baseline_trials, generate_plots):
  args = locals()

  if not os.path.exists("outputs/"):
    os.mkdir("outputs")

  if not os.path.exists("outputs/outputs_single_recovery/"):
    os.mkdir("outputs/outputs_single_recovery")

  if m >= d:
    raise Exception(f"number of segments M must be less than ambient dimension d, got M = {m}, d = {d}")

  # make subdirectory to save results in
  outdir = f"outputs/outputs_single_recovery/{exp_name}"
  if not os.path.exists(outdir):
    os.mkdir(f"outputs/outputs_single_recovery/{exp_name}")
  print(f"saving results to {outdir}")
  logfile = os.path.join(outdir, "logs")
  with open(logfile, "w") as f:
    f.write(f"logfile for experiment {exp_name}\n**********\n")

  # save arguments for posterity
  args["N"] = args.pop("n") # click annoyingly converts all args to lowercase
  args["M"] = args.pop("m")
  args["N_mini"] = args.pop("n_mini")
  with open(os.path.join(outdir, "args.json"), "w") as f:
    json.dump(args, f)
  M = m
  N = n
  N_mini = n_mini

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
    m1_raw, m2_raw, m3_raw, cloud_mini = gen_cloud_moments(subkey, C, sigma2, N, num_chunks = N // chunk_size, N_mini = N_mini)
    m1 = m1_raw
    m2 = m2_raw - sigma2 * jnp.eye(d) # de-bias second and third moments
    m3 = m3_raw - 3 * sigma2 * compute_sym_part(jnp.einsum("i, jk -> ijk", m1, jnp.eye(d)))

  # estimate the subspaces that the control points live in
  subspaces, status = recover_C_subspaces(m3, m2, M, seed = 0)
  with open(logfile, "a") as f:
    f.write(f"subspace sorting status: {status}\n")
  
  # estimate the curve from subspaces alone
  Lhat_init = jnp.ones(shape = (M+1,))
  Chat_subspaces = estimate_C_from_subspaces(subspaces, Lhat_init, m1, m2, m3, nit = length_gd_nit)

  curve_loss_subspace_estimation = compute_curve_loss(C, Chat_subspaces)
  m3_loss_subspace_estimation = m3_loss(Chat_subspaces, compute_m3(C))
  with open(logfile, "a") as f:
    f.write(f"curve loss from subspace estimation: {curve_loss_subspace_estimation:.5f}\n")
    f.write(f"third moment loss from subspace estimation: {m3_loss_subspace_estimation:.5f}\n")

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
  with open(logfile, "a") as f:
    f.write(f"curve loss from third moment finetuning: {curve_loss_m3_finetuning:.5f}\n")
    f.write(f"third moment loss from third moment finetuning: {m3_loss_m3_finetuning:.5f}\n")

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
  with open(logfile, "a") as f:
    f.write(f"curve loss from triple moment baseline: {curve_loss_m123_baseline:.5f}\n")
    f.write(f"third moment loss from triple moment baseline: {m3_loss_m123_baseline:.5f}\n")
    f.write(f"curve loss from third moment baseline: {curve_loss_m3_baseline:.5f}\n")
    f.write(f"third moment loss from third moment baseline: {m3_loss_m3_baseline:.5f}\n")

  # export recovered curve and cloud to .npz
  if use_true_moments:
    np.savez(f"{outdir}/curves", C = C, Chat_subspaces = np.array(Chat_subspaces), Chat = np.array(Chat), Chat_baseline_m3 = np.array(Chat_baseline_m3), Chat_baseline_m123 = np.array(Chat_baseline_m123))
  else:
    np.savez(f"{outdir}/curves", C = C, Chat_subspaces = np.array(Chat_subspaces), Chat = np.array(Chat), Chat_baseline_m3 = np.array(Chat_baseline_m3), Chat_baseline_m123 = np.array(Chat_baseline_m123), cloud_mini = np.array(cloud_mini))

  if generate_plots:
    idx0, idx1 = 0, 1

    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(C[:, idx0], C[:, idx1], linewidth = 2, color = "blue", label = "ground truth curve")
    ax.plot(Chat[:, idx0], Chat[:, idx1], linewidth = 2, linestyle = "dashdot", color = "orange", label = "predicted curve", alpha = 0.9)

    if not use_true_moments:
      ax.scatter(cloud_mini[:, idx0], cloud_mini[:, idx1], s = 1, color = "blue", label = "point cloud")

    x_max = np.max(np.abs(C[:, idx0]))
    y_max = np.max(np.abs(C[:, idx1]))
    l = max(x_max, y_max)
    ax.set_xlim([-l - l/5, l + l/5]) # pad with margin
    ax.set_ylim([-l - l/5, l + l/5])  
    ax.tick_params(labelsize = 10)
    ax.set_aspect("equal")
    ax.legend(fontsize = 18, loc = "upper right")
    fig.savefig(os.path.join(outdir, "plots.png"), dpi = 300)


if __name__ == "__main__":
  t0 = time.time()
  main()
  t1 = time.time()
  print(f"full experiment took {(t1 - t0):.2f} seconds")
