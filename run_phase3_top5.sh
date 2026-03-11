#!/bin/bash
# Phase 3: Run top-5 Borda features (15 conditions) with max 5 concurrent jobs.
#
# Uses a job queue: launches up to MAX_JOBS conditions in parallel, and starts
# the next condition as soon as one finishes. Skips seed-runs that already
# have >= BUDGET candidates in their log.
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
#
#   # Override max parallel jobs (default 5):
#   MAX_JOBS=3 nohup bash run_phase3_top5.sh > logs/phase3_top5.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="${PHASE3_RESULTS_DIR:-results_phase3}"
BUDGET=100
MAX_JOBS="${MAX_JOBS:-5}"

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
FORMATS=(neutral directional comparative)

# Check if a seed-run is already complete (>= BUDGET candidates in log)
is_complete() {
    local cond="$1" seed="$2"
    local seed_dir="$RESULTS_DIR/$cond/seed-$seed"
    local log
    log=$(ls "$seed_dir"/run-*/log.jsonl 2>/dev/null | head -1)
    [ -z "$log" ] && return 1
    local count
    count=$(wc -l < "$log")
    [ "$count" -ge "$BUDGET" ]
}

# Wait until fewer than MAX_JOBS are running, return when a slot opens
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
}

# Build list of (condition, seeds_to_run) pairs
declare -a QUEUE_COND=()
declare -a QUEUE_SEEDS=()

for feat in "${FEATURES[@]}"; do
    for fmt in "${FORMATS[@]}"; do
        cond="${fmt}-${feat}"

        # Check if this condition exists
        python -c "from experiments.phase3_config import get_conditions; assert '${cond}' in get_conditions()" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "[$(date '+%H:%M:%S')] SKIP $cond (not in condition registry)"
            continue
        fi

        # Check which seeds still need running
        seeds_to_run=()
        for seed in 0 1 2 3 4; do
            if is_complete "$cond" "$seed"; then
                echo "[$(date '+%H:%M:%S')] SKIP $cond seed-$seed (already complete)"
            else
                seeds_to_run+=($seed)
            fi
        done

        if [ ${#seeds_to_run[@]} -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] SKIP $cond (all seeds complete)"
            continue
        fi

        QUEUE_COND+=("$cond")
        QUEUE_SEEDS+=("${seeds_to_run[*]}")
    done
done

total=${#QUEUE_COND[@]}
echo ""
echo "Job queue: $total conditions to run, max $MAX_JOBS parallel"
echo ""

# Launch jobs with throttling
failed=0
launched=0
for i in "${!QUEUE_COND[@]}"; do
    cond="${QUEUE_COND[$i]}"
    seeds="${QUEUE_SEEDS[$i]}"

    wait_for_slot

    ((launched++))
    echo "[$(date '+%H:%M:%S')] [$launched/$total] Starting $cond  seeds=[$seeds]"
    python run_phase3.py "$cond" --seeds $seeds \
        > "logs/phase3_${cond}.log" 2>&1 &
    sleep 3  # stagger API calls
done

# Wait for remaining jobs
echo ""
echo "[$(date '+%H:%M:%S')] All $total jobs launched, waiting for stragglers..."
wait

echo ""
echo "[$(date '+%H:%M:%S')] All done. Check individual logs in logs/phase3_*.log"
