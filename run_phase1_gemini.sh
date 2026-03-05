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

if [ -f .env ]; then
    set -a && source .env && set +a
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "ERROR: GOOGLE_API_KEY not set. Add to .env or export it."
    exit 1
fi

MODELS="gemini-3-flash gemini-3-pro"
SEEDS="0 1 2 3 4"

pids=()
for model in $MODELS; do
    for seed in $SEEDS; do
        echo "[$(date '+%H:%M:%S')] Starting $model seed $seed"
        python run_phase1.py "$model" --seeds "$seed" > "logs/${model}_s${seed}.log" 2>&1 &
        pids+=($!)
    done
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
