#!/usr/bin/env bash
#SBATCH --job-name=p4_fs_smoke
#SBATCH --output=/data/s3815129/slurm_logs/p4_fs_smoke_%j.out
#SBATCH --error=/data/s3815129/slurm_logs/p4_fs_smoke_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=L40s_students
#SBATCH --nodelist=saronite

# Smoke timing for §5.4.6 full-suite generalisation.
# Sequentially runs every (algorithm, dim) combo with --instance-end 5
# (5 instances x 5 seeds = 25 runs per shard, 375 runs total) on a single
# CPU. Per-shard wall time is logged with date markers so we can project
# total compute for the full 1000-instance sweep.

set -uo pipefail

ENV=/local/$USER/conda_envs/thesis
REPO_DIR="$HOME/thesis"

cd "$REPO_DIR"

export PHASE4_FULL_SUITE_DIR="/data/$USER/results_phase4_full_suite_smoke"
mkdir -p "$PHASE4_FULL_SUITE_DIR"
mkdir -p /data/$USER/slurm_logs

ALGS=(vanilla_winner neutral_winner sage_winner combined_neutral_winner cma_es)
DIMS=(5 10 20)
INSTANCES=5

echo "=== SMOKE START $(date -Iseconds) ==="
echo "NODE=$(hostname)"
echo "PYTHON=$ENV/bin/python ($($ENV/bin/python -V 2>&1))"
echo "OUT=$PHASE4_FULL_SUITE_DIR"
echo "INSTANCES_PER_SHARD=$INSTANCES (x 5 seeds)"
echo

for ALG in "${ALGS[@]}"; do
  for DIM in "${DIMS[@]}"; do
    echo "--- SHARD START alg=$ALG dim=$DIM at $(date -Iseconds) ---"
    T0=$(date +%s)
    $ENV/bin/python run_phase4_full_suite.py \
      --algorithm "$ALG" --dim "$DIM" \
      --instance-start 0 --instance-end "$INSTANCES"
    RC=$?
    T1=$(date +%s)
    echo "--- SHARD END   alg=$ALG dim=$DIM rc=$RC dt=$((T1 - T0))s ---"
    echo
  done
done

echo "=== SMOKE FINISHED $(date -Iseconds) ==="
