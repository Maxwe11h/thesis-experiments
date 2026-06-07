#!/usr/bin/env bash
#SBATCH --job-name=p4_fs_full
#SBATCH --output=/data/s3815129/slurm_logs/p4_fs_full_%j.out
#SBATCH --error=/data/s3815129/slurm_logs/p4_fs_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=L40s_students
#SBATCH --nodelist=saronite

# Stage 4.6 full-suite generalisation, single-job driver (Option A).
# Loops every (algorithm, dim) shard sequentially; each shard runs the
# (instance x seed) grid in a multiprocessing.Pool of $SLURM_CPUS_PER_TASK.
# 5 algs x 3 dims x 1000 instances x 5 seeds = 75,000 runs total.
# Projected: ~24 CPU-h / 32 CPUs ~= 45 min wall-clock.

set -uo pipefail

ENV=/local/$USER/conda_envs/thesis
REPO_DIR="$HOME/thesis"

cd "$REPO_DIR"

export PHASE4_FULL_SUITE_DIR="/data/$USER/results_phase4_full_suite"
export EVAL_TIMEOUT_SECONDS=14400  # 4 h fallback bound (SLURM walltime is the real cap)
mkdir -p "$PHASE4_FULL_SUITE_DIR"
mkdir -p /data/$USER/slurm_logs

ALGS=(vanilla_winner neutral_winner sage_winner combined_neutral_winner cma_es)
DIMS=(5 10 20)
N_WORKERS=${SLURM_CPUS_PER_TASK:-32}

echo "=== FULL SWEEP START $(date -Iseconds) ==="
echo "NODE=$(hostname)"
echo "PYTHON=$ENV/bin/python ($($ENV/bin/python -V 2>&1))"
echo "OUT=$PHASE4_FULL_SUITE_DIR"
echo "N_WORKERS=$N_WORKERS"
echo "EVAL_TIMEOUT_SECONDS=$EVAL_TIMEOUT_SECONDS"
echo

for ALG in "${ALGS[@]}"; do
  for DIM in "${DIMS[@]}"; do
    echo "--- SHARD START alg=$ALG dim=$DIM at $(date -Iseconds) ---"
    T0=$(date +%s)
    $ENV/bin/python -m experiments.run.run_phase4_full_suite \
      --algorithm "$ALG" --dim "$DIM" \
      --instance-start 0 --instance-end 1000 \
      --n-workers "$N_WORKERS"
    RC=$?
    T1=$(date +%s)
    echo "--- SHARD END   alg=$ALG dim=$DIM rc=$RC dt=$((T1 - T0))s ---"
    echo
  done
done

echo "=== FULL SWEEP FINISHED $(date -Iseconds) ==="
