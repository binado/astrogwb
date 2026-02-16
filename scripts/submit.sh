#!/bin/bash

function usage() {
    echo "Usage: $0 [-i <injection_file>] [-n <num_jobs>] [-o <output_dir>]"
    echo "       $0 <injection_file> [num_jobs] [output_dir]"
    echo "  injection_file: Path to the file containing injection parameters (CSV format)"
    echo "  num_jobs: Optional. Number of parallel jobs to run (default: 100)"
    echo "  output_dir: Optional. Directory to save the output files (default: out)"
    exit 1
}

FILENAME=""
NUM_JOBS=""
OUTPUT_DIR="out"
POSITIONAL_ARGS=()

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -i|--input)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1."
                usage
            fi
            FILENAME="$2"
            shift 2
            ;;
        -n|--num-jobs)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1."
                usage
            fi
            NUM_JOBS="$2"
            shift 2
            ;;
        -o|--output-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1."
                usage
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        -*)
            echo "Error: Unknown option '$1'."
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${FILENAME}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    FILENAME="${POSITIONAL_ARGS[0]}"
fi
if [[ -z "${NUM_JOBS}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    NUM_JOBS="${POSITIONAL_ARGS[1]}"
fi
if [[ "${OUTPUT_DIR}" == "out" && ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
    OUTPUT_DIR="${POSITIONAL_ARGS[2]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -gt 3 ]]; then
    echo "Error: Too many positional arguments."
    usage
fi

if [[ -z "${FILENAME}" ]]; then
    echo "Error: Missing injection file."
    usage
fi

if [[ ! -f "${FILENAME}" ]]; then
    echo "Error: Injection file '${FILENAME}' not found."
    exit 1
fi

# Calculate total injections
TOTAL_INJECTIONS=$(($(wc -l < "${FILENAME}") - 1))
if [[ ${TOTAL_INJECTIONS} -le 0 ]]; then
    echo "Error: Injection file is empty or only contains a header."
    exit 1
fi

NUM_JOBS=${NUM_JOBS:-100}
if ! [[ "${NUM_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_jobs must be a positive integer (got '${NUM_JOBS}')."
    exit 1
fi
if [[ ${NUM_JOBS} -gt ${TOTAL_INJECTIONS} ]]; then
    NUM_JOBS=${TOTAL_INJECTIONS}
fi

# Calculate batch size (ceil division)
BATCH_SIZE=$(((TOTAL_INJECTIONS + NUM_JOBS - 1) / NUM_JOBS))

echo "Submitting ${NUM_JOBS} jobs for ${TOTAL_INJECTIONS} injections (batch size: ${BATCH_SIZE})"

# Use a quoted heredoc 'EOF' to prevent local variable expansion
sbatch --array=0-$((NUM_JOBS - 1)) \
--export=ALL,INJECTION_FILE="${FILENAME}",OUTPUT_DIR="${OUTPUT_DIR}",BATCH_SIZE="${BATCH_SIZE}" << 'EOF'
#!/bin/bash
#SBATCH --job-name=inj-waveforms
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

# Work split config (passed via --export)
BATCH_SIZE="${BATCH_SIZE:-5000}"
export CHUNKSIZE="${CHUNKSIZE:-100}"
export NWORKERS="${NWORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${OUTPUT_DIR:-out}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export OFFSET=$((TASK_ID * BATCH_SIZE))
export BATCH="${BATCH_SIZE}"
export OUTPUT_FILE="${OUTPUT_DIR}/waveforms_batch_${TASK_ID}.h5"

mkdir -p "${OUTPUT_DIR}"

# Avoid CPU oversubscription in BLAS/OpenMP libs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load miniconda/24.4.0-libmamba
conda activate "${CONDA_DEFAULT_ENV}"

# Choose launcher based on environment.
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  RUNNER=(python)
fi

if command -v jobnanny >/dev/null 2>&1; then
  RUNNER=(jobnanny "${RUNNER[@]}")
fi

echo "Task ${TASK_ID}: offset=${OFFSET} batch=${BATCH_SIZE} workers=${NWORKERS} output=${OUTPUT_FILE}"

"${RUNNER[@]}" scripts/generate_injection_waveforms.py
EOF
