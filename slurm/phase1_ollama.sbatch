#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --job-name=phase1
#SBATCH --output=logs/slurm/phase1-%x-%j.out
#SBATCH --error=logs/slurm/phase1-%x-%j.err
#
# Run Phase 1 for one or more Ollama models on a single GPU.
# Seeds run in PARALLEL within each model (GPU inference + CPU evaluation
# interleave naturally), then models run sequentially.
#
# Usage:
#   # Single model, all 5 seeds in parallel
#   sbatch slurm/phase1_ollama.sbatch qwen3.5-4b
#
#   # Multiple models chained on one GPU
#   sbatch slurm/phase1_ollama.sbatch "qwen3.5-4b qwen3.5-9b granite4-3b"
#
#   # Pin to specific node
#   sbatch --nodelist=ceratanium slurm/phase1_ollama.sbatch qwen3.5-27b
#
# Prerequisites:
#   - Run slurm/setup_node.sh on the target node first
#   - Thesis repo at ~/thesis with submodules

set -euo pipefail

MODELS="${1:?Usage: sbatch slurm/phase1_ollama.sbatch \"model1 [model2 ...]\"}"
SEEDS="${2:-0 1 2 3 4}"
PARALLEL_SEEDS="${3:-5}"   # max seeds to run concurrently

# --- Paths ---
OLLAMA_DIR="/local/$USER/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
export OLLAMA_MODELS="/local/$USER/ollama_models"
CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"

# Unique port per job to avoid collisions
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 100)))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

NODE=$(hostname)
echo "========================================="
echo "Phase 1 — Optimised GPU run"
echo "  Node:     $NODE"
echo "  GPU:      ${CUDA_VISIBLE_DEVICES:-auto}"
echo "  Port:     $OLLAMA_PORT"
echo "  Models:   $MODELS"
echo "  Seeds:    $SEEDS"
echo "  Parallel: $PARALLEL_SEEDS concurrent seeds"
echo "  CPUs:     $SLURM_CPUS_PER_TASK"
echo "  Job:      $SLURM_JOB_ID"
echo "  Start:    $(date)"
echo "========================================="

# --- Verify setup ---
if [ ! -x "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama not found at $OLLAMA_BIN" >&2
    echo "Run setup_node.sh on $NODE first." >&2
    exit 1
fi

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: Conda env not found at $CONDA_ENV" >&2
    exit 1
fi

# --- Start Ollama ---
echo ""
echo "Starting Ollama on port $OLLAMA_PORT..."
$OLLAMA_BIN serve > /tmp/ollama-${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
    echo ""
    echo "Cleaning up (stopping Ollama PID $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null
    wait $OLLAMA_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Wait for server
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/" > /dev/null 2>&1; then
        echo "Ollama ready (${i}s)"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Ollama failed to start" >&2
        cat /tmp/ollama-${SLURM_JOB_ID}.log >&2
        exit 1
    fi
    sleep 1
done

# --- Activate conda ---
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Python: $(python --version)"
cd "$REPO_DIR"
mkdir -p logs/slurm

# --- Run models sequentially, seeds in parallel ---
TOTAL_START=$SECONDS

for MODEL_TAG in $MODELS; do
    echo ""
    echo "========================================="
    echo "  MODEL: $MODEL_TAG"
    echo "  Time:  $(date)"
    echo "========================================="

    MODEL_START=$SECONDS
    PIDS=()
    SEED_ARRAY=($SEEDS)

    # Launch seeds in parallel, up to $PARALLEL_SEEDS at a time
    for seed in "${SEED_ARRAY[@]}"; do
        python run_phase1.py "$MODEL_TAG" \
            --seeds "$seed" \
            --port "$OLLAMA_PORT" \
            --results-dir results_phase1 \
            > "logs/slurm/phase1-${MODEL_TAG}-seed${seed}-${SLURM_JOB_ID}.log" 2>&1 &
        PIDS+=($!)
        echo "  Launched seed $seed (PID ${PIDS[-1]})"

        # Throttle if we've hit the parallel limit
        if [ ${#PIDS[@]} -ge "$PARALLEL_SEEDS" ]; then
            # Wait for any one to finish before launching more
            wait -n "${PIDS[@]}" 2>/dev/null || true
            # Clean up finished PIDs
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
        fi
    done

    # Wait for all seeds of this model to finish
    echo "  Waiting for ${#PIDS[@]} remaining seeds..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: seed process $pid exited with error"
            FAILED=$((FAILED + 1))
        fi
    done

    MODEL_ELAPSED=$(( SECONDS - MODEL_START ))
    echo "  $MODEL_TAG complete: ${MODEL_ELAPSED}s ($(( MODEL_ELAPSED / 3600 ))h $(( (MODEL_ELAPSED % 3600) / 60 ))m), $FAILED failures"

    # Generate summary CSVs
    python run_phase1.py "$MODEL_TAG" --summarise --results-dir results_phase1 2>/dev/null || true
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo ""
echo "========================================="
echo "All models complete"
echo "  Total: ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m)"
echo "  End:   $(date)"
echo "========================================="
