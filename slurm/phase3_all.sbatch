#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:0
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --job-name=p3-all
#SBATCH --output=logs/slurm/phase3-all-%j.out
#SBATCH --error=logs/slurm/phase3-all-%j.err
#
# Run all Phase 3 top-5 conditions with ALL seeds in parallel.
# Each (condition, seed) pair is its own background process.
# 15 conditions × 5 seeds = up to 75 processes (mostly API-waiting).
# No GPU needed — Gemini API only.
#
# Usage:
#   sbatch --nodelist=saronite slurm/phase3_all.sbatch

set -uo pipefail

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase3"
BUDGET=100

NODE=$(hostname)
echo "========================================="
echo "Phase 3 — All top-5 conditions (all seeds parallel)"
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

# --- Build condition list ---
CONDITIONS=(
    neutral-avg_improvement directional-avg_improvement comparative-avg_improvement
    neutral-intensification_ratio directional-intensification_ratio comparative-intensification_ratio
    neutral-fitness_plateau_fraction directional-fitness_plateau_fraction comparative-fitness_plateau_fraction
    neutral-step_size_autocorrelation directional-step_size_autocorrelation comparative-step_size_autocorrelation
    neutral-improvement_spatial_correlation directional-improvement_spatial_correlation comparative-improvement_spatial_correlation
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
