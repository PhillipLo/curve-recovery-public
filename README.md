# Method of Moments for Estimation of Noisy Curves

This is the GitHub repository for the paper [arXiv link here]. It contains all the code needed to replicate the numerical results of the paper. 

## Installation

The following instructions are for Unix/macOS only.

To install:
* Clone the repository: `git clone https://github.com/PhillipLo/curve-recovery-public.git`.
* Create a new virtual environment: `python3 -m venv /path/to/new/venv`.
* Activate the new `venv` with `source <venv>/bin/activate`, replacing `<venv>` with the location of your virtual environment. If you currently have a `conda` environment running, make sure to deactivate it with `conda deactivate`.
* Install the packages using `python3 -m pip install -r requirements-cpu.txt` if you'd like to run on CPU only, or `python3 -m pip install -r requirements-gpu.txt` if your system is NVIDIA CUDA12 enabled. Note that the large scale experiments (with 500 trials) conducted in the paper will not run on CPU in a reasonable amount of time, but single problem instances in lower dimensions are still feasible. 
* Note that JAX installation with CUDA can sometimes be finicky, and the requirements for running on GPU might differ depending on your hardware/CUDA drivers. If the installation fails, you may have to create your environment with the following packages:
  - JAX, JAXopt, click, matplotlib, pandas, tqdm, numpy, scipy, plotly
* To verify installation: `python tests.py`. Tests should run in reasonable time (a few minutes) on CPU.

## Files
* `README.md`: This file.
* `requirements-gpu.txt`, `requirements-gpu.txt`: Outputs of `pip freeze` to install into `venv`.
* `tests.py`: Automated tests.
* `data_generation.py`, `loss_functions.py`, `moments.py`, `optim.py`, `projection.py`, `relaxed_moments.py`, `subspace_recovery.py`, `tracing_recovery.py`: Contain various subroutines/helper functions.
* `moment_matching_recovery_demo.ipynb`: A Jupyter notebook demonstrating our moment matching recovery algorithm (Algorithm 3 in the paper) for a single curve and plotting results. Small enough of a problem instance to run on CPU.
* `tracing_demo.ipynb`: A Jupyter notebook demonstrating our low noise tracing-based curve recovery algorithm (from Appendix A in the paper) for a single curve and plotting results. Small enough of a problem instance to run on CPU.
* `moment_matching_recovery_experiment.py`: Full pipeline to run a moment matching experiment on a single curve instance.
* `moment_matching_recovery_experiment_batched.py`: Full pipeline to run a batch of moment matching experiments on multiple curves.

## Running moment matching algorithm experiments

### Single Experiments

To run a single experiment on matching a curve using our method of moments approach, run `python moment_matching_recovery_experiment.py`, which accepts the following command line arguments:
* `--exp_name`: name to give the experiment
* `--seed`: integer for RNG seeding, default `88`
* `--d`: ambient dimension of the curve, default `12`
* `--M`: number of segments in the curve, default `8`
* `--sigma2`: noise level, default `0.25`
* `--use_true_moments`: whether to run the experiment with ground truth access to curve moments or estimate the moments from a point cloud, default `True`
* `--N`: size of point cloud to generate if `--use_true_moments = False`, default `10_000_000`
* `--N_mini`: size of small point cloud to generate for visualization purposes only, default `1_000`
* `--chunk_size`: chunk size for chunked cloud moment computation, default `100_000`
* `--length_gd_nit`: number of gradient descent iterations to use in the minimization in the first phase of Algorithm 3, default `100`
* `--finetune_nit`: number of gradient descent iterations to use in the alternating gradient descent of the second phase of Algorithm 3, default `2500`
* `--num_baseline_trials`: number of random restarts to run in the baseline, default `10`
* `--generate_plots`: whether to generate a very basic plot of the ground truth and predicted curves, default `True`

All the default values have been chosen so that the experiment runs in a few seconds on CPU only on an M1 MacBook Pro; they are NOT the values used to generate the plots in the paper. The more computationally expensive experiment where the moments are estimated from a cloud (i.e., `--use_true_moments = False`) is also able to run on a laptop with no issue.

The results will be saved to `outputs/outputs_single_recovery/<exp_name>`, which contains:
  * `logs` containing numerical results
  * `args.json` containing the passed arguments
  * `plots.png` (if `--generate_plots = True`)
  * `curves.npz` containing the ground truth curve, predictions, and the point cloud (if applicable), which can be unpacked and plotted in your favorite plotting software.

### Batch Experiments

To run batches of experiments, run `python moment_matching_recovery_experiment_batched.py`, which accepts same command line arguments (but with different default values) as the `moment_matching_recovery_experiment.py` (except for `--generate_plots` and `--N_mini`), as well as the following additional argument:
* `--num_experiments`: number of experiments to run, default `500`

The two batch experiments in the paper were run with the following two commands (the first for the experiment with access to ground truth moments, the second with cloud-estimated moments):
```
python moment_matching_recovery_experiment_batched.py --exp_name true_batch --num_experiments 500 --seed 88 --d 48 --M 32 --use_true_moments True

python moment_matching_recovery_experiment_batched.py --exp_name cloud_batch --num_experiments 500 --seed 88 --d 24 --M 16 --sigma2 0.25 --N 200_000_000 --use_true_moments False --chunk_size 5_000_000
```
Note that these experiments will not run in reasonable time on CPU only. Our experiments were run on NVIDIA RTX A6000 GPUs with 48 GB of memory; you may need to reduce the value of the `--chunk_size` argument if your GPU is smaller. Running on multiple GPUs is not supported. The first experiment took about 3.5 hours to run, while the second took about 7 hours.

The results will be saved to `outputs/outputs_batch_recovery/<exp_name>`, which contains: 
  * `df.csv` containing losses, runtimes, and other information for each individual experiment
  * `summary_df.csv` containing the quartiles reported in the paper.
  * `args.json` containing the passed arguments
  * `results.json` containing some miscellaneous summary statistics.
    