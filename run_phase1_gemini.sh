#!/bin/bash
# Launch all Gemini Phase 1 runs in parallel from a single process.
#
# Usage:
#   conda activate /local/$USER/conda_envs/thesis
#   cd /local/$USER/thesis
#   nohup bash run_phase1_gemini.sh > logs/gemini_all.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate /local/$USER/conda_envs/thesis

if [ -f .env ]; then
    set -a && source .env && set +a
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "ERROR: GOOGLE_API_KEY not set. Add to .env or export it."
    exit 1
fi

MODELS="gemini-3-flash gemini-3-pro"

# Run 2 models in parallel, seeds sequential within each model.
# (10 parallel processes + worker pools hit the per-user process limit.)
pids=()
for model in $MODELS; do
    echo "[$(date '+%H:%M:%S')] Starting $model (all seeds sequential)"
    python run_phase1.py "$model" > "logs/${model}.log" 2>&1 &
    pids+=($!)
done

echo ""
echo "Launched ${#pids[@]} jobs: ${pids[*]}"
echo "Waiting for all to finish..."

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[$(date '+%H:%M:%S')] PID $pid FAILED"
        ((failed++))
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] All done. $failed/${#pids[@]} failed."
