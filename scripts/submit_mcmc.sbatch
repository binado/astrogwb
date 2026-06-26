#!/bin/bash
# Example SLURM job-array wrapper for scripts/run_mcmc.py.
# One TOML config file per array task: configs/task_0.toml, configs/task_1.toml, ...
# Adapt --partition / --account / module loads / env activation to your cluster.
#
#SBATCH --job-name=asgwb-mcmc
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --error=logs/mcmc_%A_%a.err
#
# GPU runs: add the line below and set platform = "gpu" (or "auto") in the config.
# The script auto-detects the device and switches chain_method to "vectorized".
##SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

# Pin CPU threads to the allocation so XLA/OMP do not oversubscribe the node.
# (run_mcmc.py also honors [runtime] cpu_threads; either is sufficient.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

CONFIG="configs/task_${SLURM_ARRAY_TASK_ID}.toml"
echo "Task ${SLURM_ARRAY_TASK_ID}: running ${CONFIG}"

# Ensure the cluster env has dev deps (arviz, jupyter, etc.): uv sync --group dev
uv run python scripts/run_mcmc.py --config "${CONFIG}"
