#!/usr/bin/env bash
#SBATCH --job-name=p4_fs_nade
#SBATCH --output=/data/s3815129/slurm_logs/p4_fs_nade_%j.out
#SBATCH --error=/data/s3815129/slurm_logs/p4_fs_nade_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=L40s_students
#SBATCH --nodelist=saronite

# Stage 4.6 full-suite: NeighborhoodAdaptiveDE only (2025 MA-BBOB competition winner).
# 1 alg x 3 dims x 1000 instances x 5 seeds = 15,000 runs. CPU-only.
set -uo pipefail

ENV=/local/$USER/conda_envs/thesis
REPO_DIR="$HOME/thesis"
cd "$REPO_DIR"

export PHASE4_FULL_SUITE_DIR="/data/$USER/results_phase4_full_suite"
export EVAL_TIMEOUT_SECONDS=14400
mkdir -p "$PHASE4_FULL_SUITE_DIR"
mkdir -p /data/$USER/slurm_logs

ALG=neighborhood_adaptive_de
DIMS=(5 10 20)
N_WORKERS=${SLURM_CPUS_PER_TASK:-32}

echo "=== NADE SWEEP START $(date -Iseconds) ==="
echo "NODE=$(hostname)"; echo "OUT=$PHASE4_FULL_SUITE_DIR"; echo "N_WORKERS=$N_WORKERS"
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
done
echo "=== NADE SWEEP FINISHED $(date -Iseconds) ==="
