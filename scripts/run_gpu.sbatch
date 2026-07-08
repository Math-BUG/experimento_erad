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
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
mkdir -p "$OUTPUT_DIR"
RESULT_DIR="$OUTPUT_DIR/gpu/$RUN_ID"
mkdir -p "$RESULT_DIR"

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
echo "GPU_MONITOR_INTERVAL=$GPU_MONITOR_INTERVAL"
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

GPU_MONITOR_PID=""
cleanup_gpu_monitor() {
  if [ -n "${GPU_MONITOR_PID:-}" ]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_gpu_monitor EXIT

start_gpu_monitor() {
  local target_pid="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  GPU_ID_SELECTOR="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
  GPU_USAGE_LOG="$RESULT_DIR/gpu_usage.csv"
  GPU_PROCESS_MEMORY_LOG="$RESULT_DIR/gpu_process_memory.csv"
  GPU_PROCESS_UTIL_LOG="$RESULT_DIR/gpu_process_utilization.csv"
  GPU_PROCESS_MEMORY_RAW_LOG="$RESULT_DIR/gpu_process_memory_raw.csv"
  GPU_MONITOR_DEBUG_LOG="$RESULT_DIR/gpu_monitor_debug.log"
  {
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "GPU_ID_SELECTOR=${GPU_ID_SELECTOR:-all-visible-to-nvidia-smi}"
    echo "PYTHON_PID=$target_pid"
    nvidia-smi -L || true
  } > "$RESULT_DIR/gpu_allocation.txt"

  (
    collect_descendants() {
      local root_pid="$1"
      local exclude_pid="$2"
      local child
      pgrep -P "$root_pid" 2>/dev/null | while read -r child; do
        if [ "$child" = "$exclude_pid" ]; then
          continue
        fi
        echo "$child"
        collect_descendants "$child" "$exclude_pid"
      done
    }

    gpu_query_args=()
    gpu_pmon_args=()
    if [ -n "$GPU_ID_SELECTOR" ]; then
      gpu_query_args=(--id="$GPU_ID_SELECTOR")
      gpu_pmon_args=(-i "$GPU_ID_SELECTOR")
    fi

    echo "timestamp,index,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c"
    echo "timestamp,gpu_uuid,pid,process_name,used_memory_mib" > "$GPU_PROCESS_MEMORY_LOG"
    echo "timestamp,gpu_uuid,pid,process_name,used_memory_mib" > "$GPU_PROCESS_MEMORY_RAW_LOG"
    echo "timestamp,gpu_index,pid,sm_percent,mem_percent,command" > "$GPU_PROCESS_UTIL_LOG"
    echo "monitor_started=$(date '+%Y-%m-%d %H:%M:%S') target_pid=$target_pid gpu_selector=${GPU_ID_SELECTOR:-none}" > "$GPU_MONITOR_DEBUG_LOG"
    while true; do
      timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
      job_pids="$(
        {
          echo "$target_pid"
          collect_descendants "$target_pid" "$BASHPID" || true
        } | sort -n | tr '\n' ',' | sed 's/,$//'
      )"
      nvidia-smi \
        "${gpu_query_args[@]}" \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        || echo "$timestamp gpu_query_failed selector=${GPU_ID_SELECTOR:-none}" >> "$GPU_MONITOR_DEBUG_LOG"

      if [ -n "$job_pids" ]; then
        echo "$timestamp job_pids=$job_pids" >> "$GPU_MONITOR_DEBUG_LOG"
        nvidia-smi \
          "${gpu_query_args[@]}" \
          --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
          --format=csv,noheader,nounits 2>/dev/null \
          | tee >(awk -F ', *' -v ts="$timestamp" '{ print ts "," $1 "," $2 "," $3 "," $4 }' >> "$GPU_PROCESS_MEMORY_RAW_LOG") \
          | awk -F ', *' -v ts="$timestamp" -v pids="$job_pids" '
              BEGIN {
                split(pids, pid_arr, ",")
                for (idx in pid_arr) keep[pid_arr[idx]] = 1
              }
              keep[$2] { print ts "," $1 "," $2 "," $3 "," $4 }
            ' >> "$GPU_PROCESS_MEMORY_LOG" || true

        nvidia-smi pmon "${gpu_pmon_args[@]}" -c 1 -s um 2>/dev/null \
          | awk -v ts="$timestamp" -v pids="$job_pids" '
              BEGIN {
                split(pids, pid_arr, ",")
                for (idx in pid_arr) keep[pid_arr[idx]] = 1
              }
              /^#/ || NF < 8 { next }
              keep[$2] { print ts "," $1 "," $2 "," $4 "," $5 "," $8 }
            ' >> "$GPU_PROCESS_UTIL_LOG" || true
      fi

      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) > "$GPU_USAGE_LOG" &
  GPU_MONITOR_PID=$!
  echo "GPU usage log: $GPU_USAGE_LOG"
  echo "GPU process memory log: $GPU_PROCESS_MEMORY_LOG"
  echo "GPU process utilization log: $GPU_PROCESS_UTIL_LOG"
  echo "GPU process memory raw log: $GPU_PROCESS_MEMORY_RAW_LOG"
}

python "${ARGS[@]}" &
PYTHON_PID=$!
start_gpu_monitor "$PYTHON_PID"
set +e
wait "$PYTHON_PID"
PYTHON_STATUS=$?
set -e
cleanup_gpu_monitor
if [ "$PYTHON_STATUS" -ne 0 ]; then
  exit "$PYTHON_STATUS"
fi

if [ -d "$RESULT_DIR" ]; then
  find "$RESULT_DIR" -maxdepth 2 -type f | sort
fi

echo "==== ERAD GPU END ===="
date
