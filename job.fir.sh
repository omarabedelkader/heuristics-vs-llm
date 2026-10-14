#!/usr/bin/env bash

# Slurm job for the Fir cluster (Digital Research Alliance of Canada).
# It runs the same benchmarks as job.oar.sh, on a 40 GB slice (MIG 3g.40gb) of an
# NVIDIA H100: the largest model (7b) fits, and slices start sooner than full GPUs.
#
# Submit it from the repository directory on Fir:
#     cd ~/projects/def-YOURPI/heuristics-vs-llm
#     sbatch --account=def-YOURPI job.fir.sh
# Follow it:   sq                     (PD = waiting, R = running)
#              tail -f heuristic-completion.<jobid>.out
# Cancel it:   scancel <jobid>

#SBATCH --job-name=heuristic-completion-benchmark
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=heuristic-completion.%j.out
#SBATCH --error=heuristic-completion.%j.err

set -euo pipefail

# Slurm runs a copy of this script from its own spool directory, so the repository
# is the directory sbatch was called from, not the directory of this file.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -f "$ROOT_DIR/job.oar.sh" ]]; then
    echo "Run sbatch from the repository directory (job.oar.sh not found in $ROOT_DIR)." >&2
    exit 1
fi

# The pipeline only needs the Python standard library.
module purge
module load StdEnv/2023 python/3.11

# Fir compute nodes have internet access, so job.oar.sh can download Ollama, the
# models and Pharo as it does on OAR. Models are kept between jobs (about 10 GB).
export RUN_DIR="${RUN_DIR:-$ROOT_DIR/slurm-runs/${SLURM_JOB_ID:-manual-$(date '+%Y%m%d-%H%M%S')}}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$ROOT_DIR/.ollama-models}"
# One benchmark worker per CPU core requested above; Ollama gets as many request slots.
export BENCHMARK_JOBS="${BENCHMARK_JOBS:-12}"

echo "Slurm job ${SLURM_JOB_ID:-manual} on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
exec bash "$ROOT_DIR/job.oar.sh" "$@"
