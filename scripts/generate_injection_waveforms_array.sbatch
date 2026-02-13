#!/bin/bash
#SBATCH --job-name=inj-waveforms
#SBATCH --array=0-0
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

# Compute per-task offsets using:
#   OFFSET = SLURM_ARRAY_TASK_ID * BATCH_SIZE
#
# Recommended submission pattern:
#   TOTAL_INJECTIONS=700000
#   BATCH_SIZE=5000
#   NTASKS=$(((TOTAL_INJECTIONS + BATCH_SIZE - 1) / BATCH_SIZE))
#   sbatch --array=0-$((NTASKS - 1)) \
#     --export=ALL,INJECTION_FILE=/path/to/injections.txt,OUTPUT_DIR=/path/to/out, \
#     WAVEFORM_APPROXIMANT=SEOBNRv5HM,SAMPLING_FREQUENCY=2048,DURATION=8,SOURCE_TYPE=BBH, \
#     BATCH_SIZE=${BATCH_SIZE} \
#     scripts/generate_injection_waveforms_array.sbatch

# Required inputs (can be passed with --export or edited here).
INJECTION_FILE="${INJECTION_FILE:-/path/to/injections.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output_dir}"

# Waveform config.
WAVEFORM_APPROXIMANT="${WAVEFORM_APPROXIMANT:-IMRPhenomPV2_NRTIDALV2}"
REFERENCE_FREQUENCY="${REFERENCE_FREQUENCY:-20.0}"
SAMPLING_FREQUENCY="${SAMPLING_FREQUENCY:-4096}"
DURATION="${DURATION:-8}"
SOURCE_TYPE="${SOURCE_TYPE:-BNS}"

# Work split config.
BATCH_SIZE="${BATCH_SIZE:-5000}"
CHUNKSIZE="${CHUNKSIZE:-1000}"
NWORKERS="${NWORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OFFSET=$((TASK_ID * BATCH_SIZE))
OUTPUT_FILE="${OUTPUT_DIR}/waveforms_batch_${TASK_ID}.h5"

mkdir -p "${OUTPUT_DIR}"

# Avoid CPU oversubscription in BLAS/OpenMP libs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Choose launcher based on environment.
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  RUNNER=(python)
fi

if command -v jobnanny >/dev/null 2>&1; then
  RUNNER=(jobnanny "${RUNNER[@]}")
fi

echo "Task ${TASK_ID}: offset=${OFFSET} batch=${BATCH_SIZE} workers=${NWORKERS}"

"${RUNNER[@]}" scripts/generate_injection_waveforms.py \
  "${INJECTION_FILE}" \
  "${OUTPUT_FILE}" \
  --waveform-approximant "${WAVEFORM_APPROXIMANT}" \
  --reference-frequency "${REFERENCE_FREQUENCY}" \
  --sampling-frequency "${SAMPLING_FREQUENCY}" \
  --duration "${DURATION}" \
  --source-type "${SOURCE_TYPE}" \
  --chunksize "${CHUNKSIZE}" \
  --nworkers "${NWORKERS}" \
  --offset "${OFFSET}" \
  --batch "${BATCH_SIZE}"
