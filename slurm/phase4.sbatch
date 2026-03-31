#!/bin/bash
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=p4
#SBATCH --output=logs/slurm/phase4-%j.out
#SBATCH --error=logs/slurm/phase4-%j.err
#
# Phase 4: Full benchmark comparison — one wave at a time.
# 500 candidates x ~50s/candidate ≈ 7h per seed.
#
# Split into 3 runs to stay within Gemini 10k RPD quota:
#   Run 1: seeds 0-3, 4 conditions = 16 seeds (8k API calls)
#   Run 2: seeds 4-6, 4 conditions = 12 seeds (6k API calls)
#   Run 3: seeds 7-9, 4 conditions = 12 seeds (6k API calls)
#
# Usage:
#   sbatch --nodelist=saronite slurm/phase4.sbatch 0 3
#   # wait for RPD reset (midnight PT), then:
#   sbatch --nodelist=saronite slurm/phase4.sbatch 4 6
#   # next day:
#   sbatch --nodelist=saronite slurm/phase4.sbatch 7 9

set -uo pipefail

SEED_START="${1:?Usage: sbatch slurm/phase4.sbatch SEED_START SEED_END}"
SEED_END="${2:?Usage: sbatch slurm/phase4.sbatch SEED_START SEED_END}"

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
RESULTS_DIR="$REPO_DIR/results_phase4"
BUDGET=500

CONDITIONS=(vanilla behavioural sage combined)

NODE=$(hostname)
echo "========================================="
echo "Phase 4 — Full Benchmark Comparison"
echo "  Node:   $NODE"
echo "  Seeds:  $SEED_START-$SEED_END"
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
    local log
    log=$(ls -t "$seed_dir"/run-*/log.jsonl 2>/dev/null | head -1)
    [ -z "$log" ] && return 1
    local count
    count=$(wc -l < "$log")
    [ "$count" -ge "$BUDGET" ]
}

# --- Launch all seeds in this wave in parallel ---
pids=()
labels=()

for seed in $(seq "$SEED_START" "$SEED_END"); do
    for cond in "${CONDITIONS[@]}"; do
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
echo "  Seeds:     $SEED_START-$SEED_END"
echo "  Completed: $((${#pids[@]} - failed))/${#pids[@]}"
echo "  Failed:    $failed"
echo "  End:       $(date)"
echo "========================================="
