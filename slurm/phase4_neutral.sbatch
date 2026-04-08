#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:0
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --job-name=p4-neutral
#SBATCH --output=logs/slurm/phase4-neutral-%j.out
#SBATCH --error=logs/slurm/phase4-neutral-%j.err
#
# Phase 4: neutral condition (AOCC + 5 neutral features)
# 10 seeds x 500 candidates, 1 CPU per seed
# Estimated: ~35s/candidate -> ~5h/seed, all finish within 1 day
#
# Usage:
#   sbatch --nodelist=saronite slurm/phase4_neutral.sbatch

set -uo pipefail

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase4"
BUDGET=500
CONDITION="neutral"

NODE=$(hostname)
echo "========================================="
echo "Phase 4 — $CONDITION condition"
echo "  Node:   $NODE"
echo "  Job:    $SLURM_JOB_ID"
echo "  Start:  $(date)"
echo "========================================="

# --- Activate conda ---
export PATH="/local/$USER/conda_envs/thesis/bin:$PATH"
echo "Python: $(python --version)"
cd "$REPO_DIR"
mkdir -p logs/slurm

# Load env (Vertex AI project config)
if [ -f .env ]; then
    set -a && source .env && set +a
fi
if [ -z "${VERTEXAI_PROJECT:-}" ]; then
    echo "ERROR: VERTEXAI_PROJECT not set. Add to ~/thesis/.env" >&2
    exit 1
fi
echo "Vertex AI: project=$VERTEXAI_PROJECT location=${VERTEXAI_LOCATION:-global}"

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

# --- Launch all 10 seeds ---
pids=()
labels=()

for seed in 0 1 2 3 4 5 6 7 8 9; do
    if is_complete "$CONDITION" "$seed"; then
        echo "SKIP $CONDITION seed-$seed (already complete)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] Starting $CONDITION seed-$seed"
    python run_phase4.py "$CONDITION" --seeds $seed --skip-complete \
        > "logs/slurm/phase4-${CONDITION}-s${seed}-${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
    labels+=("$CONDITION/seed-$seed")
    sleep 2
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
