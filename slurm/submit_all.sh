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
#   GPU 0: 4B + 9B + 3B        ≈  8h  (3 small models)
#   GPU 1: 8B + 7B             ≈  6h  (2 medium models)
#   GPU 2: 24B + 27B           ≈  8h  (2 large models)
#   GPU 3: 30B                 ≈  5h  (1 large model)
#   Total wall time: ~8h
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

# Group models by VRAM to balance GPU time across 4 jobs
GPU0_MODELS="qwen3.5-4b qwen3.5-9b granite4-3b"    # ~3+6+2 = 11 GB peak
GPU1_MODELS="rnj-1-8b olmo3-7b"                      # ~5+5 = 10 GB peak
GPU2_MODELS="devstral-small-2-24b qwen3.5-27b"       # ~15+17 = 17 GB peak (sequential)
GPU3_MODELS="olmo3-32b"                               # ~19 GB peak

GROUPS=("$GPU0_MODELS" "$GPU1_MODELS" "$GPU2_MODELS" "$GPU3_MODELS")
NAMES=("small-a" "small-b" "large-a" "large-b")

for i in "${!GROUPS[@]}"; do
    MODELS="${GROUPS[$i]}"
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
