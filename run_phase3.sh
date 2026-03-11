#!/bin/bash
# Launch all Phase 3 conditions in parallel (each condition runs seeds sequentially).
#
# Usage:
#   conda activate /local/$USER/conda_envs/thesis
#   cd /local/$USER/thesis
#   nohup bash run_phase3.sh > logs/phase3_all.log 2>&1 &

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

mkdir -p logs

# Get all condition tags
CONDITIONS=$(python -c "
from experiments.phase3_config import get_conditions
for tag in get_conditions():
    print(tag)
")

pids=()
for cond in $CONDITIONS; do
    echo "[$(date '+%H:%M:%S')] Starting $cond"
    python run_phase3.py "$cond" > "logs/phase3_${cond}.log" 2>&1 &
    pids+=($!)
    sleep 5  # stagger API calls
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
