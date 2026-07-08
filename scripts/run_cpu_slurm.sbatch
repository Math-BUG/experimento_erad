#!/bin/bash
#SBATCH --job-name=erad_cpu_slurm
#SBATCH --partition=scientific
#SBATCH --qos=scientific-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=erad_cpu_slurm_%j.out
#SBATCH --error=erad_cpu_slurm_%j.err

set -euo pipefail

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES before override: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"

if command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load GCCcore/12.2.0 || true
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR"
echo "project_dir: $(pwd)"

if [ -n "${VENV_PATH:-}" ] && [ -d "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "WARNING: no Python virtual environment activated"
fi

export IDS_COMPUTE_BACKEND=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

DATA_PATH="${DATA_PATH:-$PROJECT_DIR/data/train_test_network.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dados/erad_experiment_results}"
RUN_ID="${RUN_ID:-cpu_slurm_${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUTPUT_DIR"

ARGS=(
  run_experiment.py
  --backend cpu
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --n-jobs "${SLURM_CPUS_PER_TASK:-24}"
)

if [ -n "${N_SAMPLES:-}" ]; then
  ARGS+=(--n-samples "$N_SAMPLES")
fi

echo "==== ERAD CPU SLURM START ===="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RUN_ID=$RUN_ID"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-24}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-empty}"
python --version

python - <<'PY'
import importlib.util

required = ["pandas", "sklearn", "pyarrow"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python packages: {', '.join(missing)}")
PY

python "${ARGS[@]}"

echo "==== ERAD CPU SLURM END ===="
date
