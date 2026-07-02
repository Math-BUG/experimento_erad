#!/bin/bash
#SBATCH --job-name=erad_gpu
#SBATCH --partition=scientific
#SBATCH --qos=scientific-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=erad_gpu_%j.out
#SBATCH --error=erad_gpu_%j.err

set -euo pipefail

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"

module --force purge
module load GCCcore/12.2.0
module load CUDA/12.6.0

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
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

export IDS_COMPUTE_BACKEND=gpu
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

DATA_PATH="${DATA_PATH:-$PROJECT_DIR/data/train_test_network.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dados/erad_experiment_results}"
RUN_ID="${RUN_ID:-gpu_${SLURM_JOB_ID:-manual}}"
GPU_MODE="${GPU_MODE:-accelerated}"
GPU_MAX_CATEGORIES_PER_COL="${GPU_MAX_CATEGORIES_PER_COL:-64}"
mkdir -p "$OUTPUT_DIR"

ARGS=(
  run_experiment.py
  --backend gpu
  --gpu-mode "$GPU_MODE"
  --gpu-max-categories-per-col "$GPU_MAX_CATEGORIES_PER_COL"
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --n-jobs "${SLURM_CPUS_PER_TASK:-32}"
)

if [ -n "${N_SAMPLES:-}" ]; then
  ARGS+=(--n-samples "$N_SAMPLES")
fi

echo "==== ERAD GPU START ===="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RUN_ID=$RUN_ID"
echo "GPU_MODE=$GPU_MODE"
echo "GPU_MAX_CATEGORIES_PER_COL=$GPU_MAX_CATEGORIES_PER_COL"
python --version

python - <<'PY'
import importlib.util
import os

required = ["pandas", "sklearn", "pyarrow"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python packages: {', '.join(missing)}")

gpu_required = ["cudf", "cuml", "cupy"]
missing_gpu = [name for name in gpu_required if importlib.util.find_spec(name) is None]
for name in gpu_required:
    print(f"{name}_available={name not in missing_gpu}")

allow_fallback = os.environ.get("ERAD_ALLOW_GPU_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
if missing_gpu and not allow_fallback:
    raise SystemExit(
        "Missing GPU packages: "
        + ", ".join(missing_gpu)
        + ". Install requirements-gpu-cu12.txt or set ERAD_ALLOW_GPU_FALLBACK=1 to run with recorded CPU fallbacks."
    )
PY

python "${ARGS[@]}"

RESULT_DIR="$OUTPUT_DIR/gpu/$RUN_ID"
if [ -d "$RESULT_DIR" ]; then
  find "$RESULT_DIR" -maxdepth 2 -type f | sort
fi

echo "==== ERAD GPU END ===="
date
