#!/bin/bash
# Submit a SLURM job array over MCMC configs in a directory.
# One array task per *.toml or *.json file (sorted); each task runs scripts/run_mcmc.py.
# Adapt --partition / --account / module loads inside the heredoc to your cluster.
#
# Usage:
#   ./scripts/submit_mcmc.sh -i configs/mcmc/sweep
#
# Ensure the cluster env has dev deps (arviz, jupyter, etc.): uv sync --group dev

set -euo pipefail

function usage() {
    echo "Usage: $0 -i <config_dir>"
    echo "       $0 <config_dir>"
    echo "  config_dir: Directory containing one *.toml or *.json config per array task"
    exit 1
}

INPUT_DIR=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -i|--input)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1." >&2
                usage
            fi
            INPUT_DIR="$2"
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
            echo "Error: Unknown option '$1'." >&2
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${INPUT_DIR}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    INPUT_DIR="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    echo "Error: Too many positional arguments." >&2
    usage
fi

if [[ -z "${INPUT_DIR}" ]]; then
    echo "Error: Missing config directory." >&2
    usage
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Error: Config directory '${INPUT_DIR}' not found." >&2
    exit 1
fi

CONFIGS=()
while IFS= read -r config; do
    CONFIGS+=("$config")
done < <(find "${INPUT_DIR}" -maxdepth 1 -type f \( -name '*.toml' -o -name '*.json' \) | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Error: No *.toml or *.json configs found in '${INPUT_DIR}'." >&2
    exit 1
fi

mkdir -p logs
MANIFEST="logs/mcmc_manifest_${RANDOM}.txt"
printf '%s\n' "${CONFIGS[@]}" > "${MANIFEST}"

ARRAY_MAX=$(( ${#CONFIGS[@]} - 1 ))
echo "Submitting ${#CONFIGS[@]} MCMC jobs (array 0-${ARRAY_MAX})"
echo "Manifest: ${MANIFEST}"
for i in "${!CONFIGS[@]}"; do
    printf '  [%s] %s\n' "$i" "${CONFIGS[$i]}"
done

# Quoted heredoc prevents local variable expansion inside the job script.
sbatch --array=0-"${ARRAY_MAX}" \
    --export=ALL,CONFIG_MANIFEST="${MANIFEST}" << 'EOF'
#!/bin/bash
#SBATCH --job-name=asgwb-mcmc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --error=logs/mcmc_%A_%a.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${CONFIG_MANIFEST}" != /* ]]; then
    CONFIG_MANIFEST="${SUBMIT_DIR}/${CONFIG_MANIFEST}"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG=$(sed -n "$((TASK_ID + 1))p" "${CONFIG_MANIFEST}")
if [[ -z "${CONFIG}" ]]; then
    echo "Error: No config for array task ${TASK_ID} in ${CONFIG_MANIFEST}" >&2
    exit 1
fi
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${SUBMIT_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config file '${CONFIG}' not found." >&2
    exit 1
fi

cd "${SUBMIT_DIR}"

# Pin CPU threads to the allocation so XLA/OMP do not oversubscribe the node.
# (run_mcmc.py also honors [runtime] cpu_threads; either is sufficient.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export INPUT="*"
export OUTPUT="*"
export JAX_PLATFORMS="cuda"

echo "Task ${TASK_ID}: running ${CONFIG}"

if command -v job-nanny >/dev/null 2>&1; then
    job-nanny uv run python scripts/run_mcmc.py --config "${CONFIG}"
else
    uv run python scripts/run_mcmc.py --config "${CONFIG}"
fi
EOF
