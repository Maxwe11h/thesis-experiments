#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=p4
#SBATCH --output=logs/slurm/phase4-%j.out
#SBATCH --error=logs/slurm/phase4-%j.err
#
# Phase 4: Full benchmark comparison.
# 500 candidates x ~3min/candidate (vanilla/behav) or ~9min/candidate (sage/combined)
#
# Seed allocation (30 seeds total, ~8.5k RPD):
#   vanilla:     seeds 0-4  (5 seeds,  ~1.1 days)
#   behavioural: seeds 0-4  (5 seeds,  ~1.1 days)
#   sage:        seeds 0-9  (10 seeds, ~2.5 days)
#   combined:    seeds 0-9  (10 seeds, ~2.5 days)
#
# Usage:
#   sbatch --nodelist=saronite slurm/phase4.sbatch

set -uo pipefail

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase4"
BUDGET=500

NODE=$(hostname)
echo "========================================="
echo "Phase 4 — Full Benchmark Comparison"
echo "  Node:   $NODE"
echo "  Job:    $SLURM_JOB_ID"
echo "  Start:  $(date)"
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

# --- Check if a seed-run is already complete ---
is_complete() {
    local cond="$1" seed="$2"
    local seed_dir="$RESULTS_DIR/$cond/seed-$seed"
    local total=0
    for log in "$seed_dir"/run-*/log.jsonl; do
        [ -f "$log" ] || continue
        local count
        count=$(wc -l < "$log")
        total=$((total + count))
    done
    [ "$total" -ge "$BUDGET" ]
}

# --- Seed assignments per condition ---
declare -A SEED_RANGES
SEED_RANGES[vanilla]="0 1 2 3 4"
SEED_RANGES[behavioural]="0 1 2 3 4"
SEED_RANGES[sage]="0 1 2 3 4 5 6 7 8 9"
SEED_RANGES[combined]="0 1 2 3 4 5 6 7 8 9"

# --- Launch all seeds ---
pids=()
labels=()

for cond in vanilla behavioural sage combined; do
    for seed in ${SEED_RANGES[$cond]}; do
        if is_complete "$cond" "$seed"; then
            echo "SKIP $cond seed-$seed (already complete)"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] Starting $cond seed-$seed"
        python run_phase4.py "$cond" --seeds $seed --skip-complete \
            > "logs/slurm/phase4-${cond}-s${seed}-${SLURM_JOB_ID}.log" 2>&1 &
        pids+=($!)
        labels+=("$cond/seed-$seed")
        sleep 2
    done
done

echo ""
echo "Launched ${#pids[@]} processes"
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
echo "  Completed: $((${#pids[@]} - failed))/${#pids[@]}"
echo "  Failed:    $failed"
echo "  End:       $(date)"
echo "========================================="
