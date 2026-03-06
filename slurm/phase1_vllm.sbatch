#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --job-name=phase1v
#SBATCH --output=logs/slurm/phase1-%x-%j.out
#SBATCH --error=logs/slurm/phase1-%x-%j.err
#
# Run Phase 1 for one or more models using vLLM on a single GPU.
# vLLM batches concurrent requests from parallel seeds automatically,
# unlike Ollama which serializes them.
#
# Usage:
#   # tag:hf_id pairs (recommended — set by submit_all.sh)
#   sbatch slurm/phase1_vllm.sbatch "qwen3.5-4b:Qwen/Qwen3.5-4B"
#
#   # Multiple models chained on one GPU
#   sbatch slurm/phase1_vllm.sbatch "qwen3.5-4b:Qwen/Qwen3.5-4B qwen3.5-9b:Qwen/Qwen3.5-9B"
#
# Prerequisites:
#   - Run slurm/setup_node.sh on the target node first
#   - Thesis repo at ~/thesis with submodules
#
# Notes:
#   - Qwen3.5 models are multimodal; we pass --limit-mm-per-prompt to
#     disable the vision encoder and save VRAM.
#   - Requires vLLM nightly for Qwen3.5 support (pip install vllm --pre)
#   - HuggingFace model IDs are in experiments/phase1_config.py VLLM_HF_MODELS

set -euo pipefail

MODELS="${1:?Usage: sbatch slurm/phase1_vllm.sbatch \"tag:hf_id [tag2:hf_id2 ...]\"}"
SEEDS="${2:-0 1 2 3 4}"
PARALLEL_SEEDS="${3:-5}"   # max seeds to run concurrently

# --- Paths ---
CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
export HF_HOME="/local/$USER/huggingface"

# Unique port per job to avoid collisions
VLLM_PORT=$((8000 + (SLURM_JOB_ID % 100)))
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

# Models that need --limit-mm-per-prompt (multimodal arch, text-only use)
MULTIMODAL_MODELS="Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B"

NODE=$(hostname)
echo "========================================="
echo "Phase 1 — vLLM GPU run"
echo "  Node:     $NODE"
echo "  GPU:      ${CUDA_VISIBLE_DEVICES:-auto}"
echo "  Port:     $VLLM_PORT"
echo "  Base URL: $VLLM_BASE_URL"
echo "  Models:   $MODELS"
echo "  Seeds:    $SEEDS"
echo "  Parallel: $PARALLEL_SEEDS concurrent seeds"
echo "  CPUs:     $SLURM_CPUS_PER_TASK"
echo "  Job:      $SLURM_JOB_ID"
echo "  Start:    $(date)"
echo "========================================="

# --- Verify setup ---
if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: Conda env not found at $CONDA_ENV" >&2
    exit 1
fi

# --- Activate conda ---
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Python: $(python --version)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
cd "$REPO_DIR"
mkdir -p logs/slurm

# --- Run models sequentially, seeds in parallel ---
TOTAL_START=$SECONDS

for MODEL_ENTRY in $MODELS; do
    # Support tag:hf_id format, or bare hf_id (tag derived from model name)
    if [[ "$MODEL_ENTRY" == *":"* ]]; then
        MODEL_TAG="${MODEL_ENTRY%%:*}"
        HF_MODEL="${MODEL_ENTRY#*:}"
    else
        HF_MODEL="$MODEL_ENTRY"
        # Derive tag from HF model name: "Qwen/Qwen3.5-4B" -> "qwen3.5-4b"
        MODEL_TAG=$(echo "${HF_MODEL##*/}" | tr '[:upper:]' '[:lower:]')
    fi

    echo ""
    echo "========================================="
    echo "  MODEL:  $MODEL_TAG"
    echo "  HF ID:  $HF_MODEL"
    echo "  Time:   $(date)"
    echo "========================================="

    # --- Build vLLM serve command ---
    VLLM_ARGS=(
        --model "$HF_MODEL"
        --port "$VLLM_PORT"
        --disable-log-requests
        --max-model-len 8192       # We only need short context for code gen
    )

    # Multimodal models: override to text-only architecture to save VRAM
    if echo "$MULTIMODAL_MODELS" | grep -qF "$HF_MODEL"; then
        echo "  [info] Multimodal model — overriding to Qwen3NextForCausalLM"
        VLLM_ARGS+=(--hf-overrides '{"architectures": ["Qwen3NextForCausalLM"]}')
    fi

    # --- Start vLLM server for this model ---
    echo "Starting vLLM server for $HF_MODEL on port $VLLM_PORT..."
    echo "  Command: python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS[*]}"
    python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
        > "/tmp/vllm-${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    # Wait for vLLM to be alive (/health) then model ready (/v1/models)
    VLLM_READY=0
    for i in $(seq 1 300); do
        # Phase 1: check server process is alive
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process died during startup" >&2
            cat "/tmp/vllm-${SLURM_JOB_ID}.log" >&2
            exit 1
        fi
        # Phase 2: check model is loaded and ready to serve
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
            echo "  vLLM model ready (${i}s)"
            VLLM_READY=1
            break
        fi
        # Print progress every 30s
        if (( i % 30 == 0 )); then
            echo "  Still loading model... (${i}s)"
        fi
        sleep 1
    done

    if [ "$VLLM_READY" -eq 0 ]; then
        echo "ERROR: vLLM failed to load model within 300s" >&2
        echo "--- vLLM log (last 50 lines) ---" >&2
        tail -50 "/tmp/vllm-${SLURM_JOB_ID}.log" >&2
        kill "$VLLM_PID" 2>/dev/null
        exit 1
    fi

    MODEL_START=$SECONDS
    PIDS=()
    SEED_ARRAY=($SEEDS)

    # Launch seeds in parallel, up to $PARALLEL_SEEDS at a time
    for seed in "${SEED_ARRAY[@]}"; do
        python run_phase1.py "$MODEL_TAG" \
            --seeds "$seed" \
            --custom-vllm "$HF_MODEL" \
            --vllm-base-url "$VLLM_BASE_URL" \
            --results-dir results_phase1 \
            > "logs/slurm/phase1-${MODEL_TAG}-seed${seed}-${SLURM_JOB_ID}.log" 2>&1 &
        PIDS+=($!)
        echo "  Launched seed $seed (PID ${PIDS[-1]})"

        # Throttle if we've hit the parallel limit
        if [ ${#PIDS[@]} -ge "$PARALLEL_SEEDS" ]; then
            wait -n "${PIDS[@]}" 2>/dev/null || true
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

    # Stop vLLM server before loading the next model
    echo "  Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true

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
