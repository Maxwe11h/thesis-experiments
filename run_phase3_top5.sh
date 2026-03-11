#!/bin/bash
# Phase 3: Run top-5 Borda features (15 conditions) with 5 parallel processes.
#
# Groups conditions by feature so each process handles all 3 formats
# (neutral, directional, comparative) sequentially. This keeps total
# concurrent processes at 5 + their worker subprocesses (~10 total),
# avoiding the school's per-user process/thread limits.
#
# Top 5 by Borda count:
#   1. avg_improvement           (Borda: 3)
#   2. intensification_ratio     (Borda: 13)
#   3. fitness_plateau_fraction  (Borda: 15)
#   4. step_size_autocorrelation (Borda: 18)
#   5. improvement_spatial_correlation (Borda: 28)
#
# Usage:
#   conda activate /local/$USER/conda_envs/thesis
#   cd /local/$USER/thesis
#   nohup bash run_phase3_top5.sh > logs/phase3_top5.log 2>&1 &

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

FEATURES=(
    avg_improvement
    intensification_ratio
    fitness_plateau_fraction
    step_size_autocorrelation
    improvement_spatial_correlation
)

pids=()
for feat in "${FEATURES[@]}"; do
    echo "[$(date '+%H:%M:%S')] Starting feature group: $feat (neutral + directional + comparative)"
    python run_phase3.py \
        "neutral-${feat}" \
        "directional-${feat}" \
        "comparative-${feat}" \
        > "logs/phase3_${feat}.log" 2>&1 &
    pids+=($!)
    sleep 3  # stagger API calls slightly
done

echo ""
echo "Launched ${#pids[@]} feature groups (15 conditions total): ${pids[*]}"
echo "Monitor with: tail -f logs/phase3_<feature>.log"
echo "Waiting for all to finish..."

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[$(date '+%H:%M:%S')] FAILED: ${FEATURES[$i]} (PID ${pids[$i]})"
        ((failed++))
    else
        echo "[$(date '+%H:%M:%S')] Done: ${FEATURES[$i]}"
    fi
done

echo ""
if [ $failed -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] All 5 feature groups completed successfully."
else
    echo "[$(date '+%H:%M:%S')] Finished with $failed/${#pids[@]} failures."
fi
