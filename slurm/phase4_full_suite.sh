#!/usr/bin/env bash
#SBATCH --job-name=p4_fs
#SBATCH --output=logs/slurm/p4_fs_%A_%a.out
#SBATCH --error=logs/slurm/p4_fs_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29       # 5 algorithms x 3 dims x 2 instance batches = 30 shards

set -euo pipefail

# 5 algorithms x 3 dims x 2 batches (0-499, 500-999) = 30 task IDs.
ALGS=(vanilla_winner neutral_winner sage_winner combined_neutral_winner cma_es)
DIMS=(5 10 20)
BATCHES=(0 500)
BATCH_SIZE=500

idx=$SLURM_ARRAY_TASK_ID
alg_i=$(( idx / 6 ))
dim_i=$(( (idx / 2) % 3 ))
batch_i=$(( idx % 2 ))

ALG=${ALGS[$alg_i]}
DIM=${DIMS[$dim_i]}
START=${BATCHES[$batch_i]}
END=$(( START + BATCH_SIZE ))

CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"

mkdir -p logs/slurm
python -m experiments.run.run_phase4_full_suite \
  --algorithm "$ALG" --dim "$DIM" \
  --instance-start "$START" --instance-end "$END"
