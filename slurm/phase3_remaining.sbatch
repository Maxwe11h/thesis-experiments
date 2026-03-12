#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:0
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --job-name=p3-rest
#SBATCH --output=logs/slurm/phase3-rest-%j.out
#SBATCH --error=logs/slurm/phase3-rest-%j.err
#
# Run remaining 14 Phase 3 conditions (bottom-5 features) with ALL seeds in parallel.
# 14 conditions × 5 seeds = up to 70 processes (mostly API-waiting).
# No GPU needed — Gemini API only.
#
# Features: half_convergence_time, fitness_autocorrelation, x_spread_early,
#           longest_no_improvement_streak (no comparative), dimension_convergence_heterogeneity
#
# Usage:
#   sbatch --nodelist=saronite slurm/phase3_remaining.sbatch

set -uo pipefail

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase3"
BUDGET=100

NODE=$(hostname)
echo "========================================="
echo "Phase 3 — Remaining 14 conditions (all seeds parallel)"
echo "  Node:  $NODE"
echo "  CPUs:  $SLURM_CPUS_PER_TASK"
echo "  Job:   $SLURM_JOB_ID"
echo "  Start: $(date)"
echo "========================================="

# --- Activate conda ---
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Python: $(python --version)"
cd "$REPO_DIR"
mkdir -p logs/slurm

# Load API key
if [ -f .env ]; then
    set -a && source .env && set +a
fi
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "ERROR: GOOGLE_API_KEY not set. Add to ~/thesis/.env" >&2
    exit 1
fi

# --- Build condition list (remaining 5 features, 14 conditions) ---
# longest_no_improvement_streak excluded from comparative (U-shaped AOCC relationship)
CONDITIONS=(
    neutral-half_convergence_time directional-half_convergence_time comparative-half_convergence_time
    neutral-fitness_autocorrelation directional-fitness_autocorrelation comparative-fitness_autocorrelation
    neutral-x_spread_early directional-x_spread_early comparative-x_spread_early
    neutral-longest_no_improvement_streak directional-longest_no_improvement_streak
    neutral-dimension_convergence_heterogeneity directional-dimension_convergence_heterogeneity comparative-dimension_convergence_heterogeneity
)

# --- Check if a seed-run is already complete ---
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

# --- Launch every (condition, seed) pair in parallel ---
pids=()
labels=()

for cond in "${CONDITIONS[@]}"; do
    for seed in 0 1 2 3 4; do
        if is_complete "$cond" "$seed"; then
            echo "SKIP $cond seed-$seed (already complete)"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] Starting $cond seed-$seed"
        python run_phase3.py "$cond" --seeds $seed \
            > "logs/slurm/phase3-${cond}-s${seed}-${SLURM_JOB_ID}.log" 2>&1 &
        pids+=($!)
        labels+=("$cond/seed-$seed")
        sleep 1  # stagger API calls
    done
done

echo ""
echo "Launched ${#pids[@]} processes (condition × seed pairs)"
echo ""

# --- Wait for all ---
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[$(date '+%H:%M:%S')] FAILED: ${labels[$i]}"
        ((failed++))
    else
        echo "[$(date '+%H:%M:%S')] Done: ${labels[$i]}"
    fi
done

echo ""
echo "========================================="
echo "  Completed: $((${#pids[@]} - failed))/${#pids[@]} seed-runs"
echo "  Failed:    $failed"
echo "  End:       $(date)"
echo "========================================="
