#!/bin/bash
# Submit Phase 1 Ollama jobs across SLURM GPUs, optimised for throughput.
#
# Strategy:
#   - 4 jobs = 4 GPUs on one node
#   - Models grouped by VRAM size so each GPU finishes around the same time
#   - Seeds run in parallel within each job (GPU inference + CPU eval interleave)
#   - Models within a job run sequentially (no model-swapping overhead)
#
# Estimated times (L40S, 5 parallel seeds):
#   GPU 0: 32B                 ≈  6h  (submitted first)
#   GPU 1: 27B + 3B            ≈  7h
#   GPU 2: 24B + 4B            ≈  7h
#   GPU 3: 8B + 7B + 9B        ≈ 10h  (submitted last)
#   Total wall time: ~10h
#
# Usage:
#   bash slurm/submit_all.sh              # defaults to ceratanium
#   bash slurm/submit_all.sh saronite     # use different node
#   DRY_RUN=1 bash slurm/submit_all.sh    # print commands without submitting

set -euo pipefail

NODE="${1:-ceratanium}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"
mkdir -p logs/slurm

echo "Phase 1 — SLURM submission plan"
echo "  Node: $NODE"
echo ""

# Large models get dedicated GPUs, submitted first for earliest start.
# Small models packed onto remaining GPU.
GPU0_MODELS="olmo3-32b"                               # ~19 GB, ~6h
GPU1_MODELS="qwen3.5-27b granite4-3b"                 # ~17+2 GB, ~7h
GPU2_MODELS="devstral-small-2-24b qwen3.5-4b"         # ~15+3 GB, ~7h
GPU3_MODELS="rnj-1-8b olmo3-7b qwen3.5-9b"            # ~5+5+6 GB, ~10h

GPU_GROUPS=("$GPU0_MODELS" "$GPU1_MODELS" "$GPU2_MODELS" "$GPU3_MODELS")
NAMES=("large-a" "large-b" "large-c" "small")

for i in "${!GPU_GROUPS[@]}"; do
    MODELS="${GPU_GROUPS[$i]}"
    NAME="${NAMES[$i]}"
    CMD="sbatch --nodelist=$NODE --job-name=p1-${NAME} slurm/phase1_ollama.sbatch \"$MODELS\""

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $CMD"
    else
        JOB_ID=$(sbatch --parsable \
            --nodelist="$NODE" \
            --job-name="p1-${NAME}" \
            slurm/phase1_ollama.sbatch "$MODELS")
        echo "  GPU $i [$NAME]: $MODELS -> job $JOB_ID"
    fi
done

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/phase1-p1-*-*.out"
echo ""
echo "Per-seed logs:"
echo "  tail -f logs/slurm/phase1-<model>-seed<N>-<jobid>.log"
