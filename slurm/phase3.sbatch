#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=p3
#SBATCH --output=logs/slurm/phase3-%x-%j.out
#SBATCH --error=logs/slurm/phase3-%x-%j.err
#
# Run one Phase 3 condition (all 5 seeds sequentially).
# No GPU needed — uses Gemini API only.
#
# Usage:
#   sbatch slurm/phase3.sbatch neutral-avg_improvement
#   sbatch slurm/phase3.sbatch comparative-step_size_autocorrelation
#
# With specific seeds:
#   sbatch slurm/phase3.sbatch neutral-avg_improvement "1 2 3 4"

set -euo pipefail

CONDITION="${1:?Usage: sbatch slurm/phase3.sbatch CONDITION [SEEDS]}"
SEEDS="${2:-0 1 2 3 4}"

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase3"
BUDGET=100

NODE=$(hostname)
echo "========================================="
echo "Phase 3 — Behavioral Feedback Screening"
echo "  Node:      $NODE"
echo "  Condition: $CONDITION"
echo "  Seeds:     $SEEDS"
echo "  Job:       $SLURM_JOB_ID"
echo "  Start:     $(date)"
echo "========================================="

# --- Activate conda ---
eval "$(conda shell.bash hook)"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
else
    # Fallback: try shared home conda env
    conda activate ~/conda_envs/thesis 2>/dev/null || {
        echo "ERROR: No conda env found. Run setup_node.sh first." >&2
        exit 1
    }
fi

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

# --- Skip completed seeds ---
SEEDS_TO_RUN=()
for seed in $SEEDS; do
    seed_dir="$RESULTS_DIR/$CONDITION/seed-$seed"
    log=$(ls "$seed_dir"/run-*/log.jsonl 2>/dev/null | head -1)
    if [ -n "$log" ] && [ "$(wc -l < "$log")" -ge "$BUDGET" ]; then
        echo "SKIP seed-$seed (already complete)"
    else
        SEEDS_TO_RUN+=($seed)
    fi
done

if [ ${#SEEDS_TO_RUN[@]} -eq 0 ]; then
    echo "All seeds already complete. Nothing to do."
    exit 0
fi

echo "Seeds to run: ${SEEDS_TO_RUN[*]}"
echo ""

# --- Run ---
START=$SECONDS

python run_phase3.py "$CONDITION" --seeds ${SEEDS_TO_RUN[*]}

ELAPSED=$(( SECONDS - START ))
echo ""
echo "========================================="
echo "  Condition: $CONDITION"
echo "  Total:     ${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"
echo "  End:       $(date)"
echo "========================================="
