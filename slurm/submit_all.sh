#!/bin/bash
# Submit Phase 1 jobs across SLURM GPUs, optimised for throughput.
#
# Strategy:
#   - 4 jobs = 4 GPUs on one node
#   - Models grouped by VRAM size so each GPU finishes around the same time
#   - Seeds run in parallel within each job (vLLM batches concurrent requests)
#   - Models within a job run sequentially (no model-swapping overhead)
#
# Backend:
#   - Default: vLLM (batches concurrent inference from parallel seeds)
#   - Fallback: Ollama (use BACKEND=ollama)
#
# Usage:
#   bash slurm/submit_all.sh              # defaults to ceratanium, vLLM
#   bash slurm/submit_all.sh saronite     # use different node
#   BACKEND=ollama bash slurm/submit_all.sh  # use Ollama backend
#   DRY_RUN=1 bash slurm/submit_all.sh    # print commands without submitting

set -euo pipefail

NODE="${1:-ceratanium}"
DRY_RUN="${DRY_RUN:-0}"
BACKEND="${BACKEND:-vllm}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"
mkdir -p logs/slurm

echo "Phase 1 — SLURM submission plan"
echo "  Node:    $NODE"
echo "  Backend: $BACKEND"
echo ""

if [ "$BACKEND" = "vllm" ]; then
    SBATCH_SCRIPT="slurm/phase1_vllm.sbatch"

    # vLLM uses HuggingFace model IDs with tag:hf_id format
    # See experiments/phase1_config.py VLLM_HF_MODELS for canonical source
    GPU0_MODELS="olmo3-32b:allenai/Olmo-3-32B-Think"
    GPU1_MODELS="qwen3.5-27b:Qwen/Qwen3.5-27B granite4-3b:ibm-granite/granite-4.0-micro"
    GPU2_MODELS="devstral-small-2-24b:mistralai/Devstral-Small-2-24B-Instruct-2512 qwen3.5-4b:Qwen/Qwen3.5-4B"
    GPU3_MODELS="rnj-1-8b:EssentialAI/rnj-1-instruct olmo3-7b:allenai/Olmo-3-7B-Instruct qwen3.5-9b:Qwen/Qwen3.5-9B"
else
    SBATCH_SCRIPT="slurm/phase1_ollama.sbatch"

    # Ollama uses its own model tags
    GPU0_MODELS="olmo3-32b"
    GPU1_MODELS="qwen3.5-27b granite4-3b"
    GPU2_MODELS="devstral-small-2-24b qwen3.5-4b"
    GPU3_MODELS="rnj-1-8b olmo3-7b qwen3.5-9b"
fi

GPU_GROUPS=("$GPU0_MODELS" "$GPU1_MODELS" "$GPU2_MODELS" "$GPU3_MODELS")
NAMES=("large-a" "large-b" "large-c" "small")

for i in "${!GPU_GROUPS[@]}"; do
    MODELS="${GPU_GROUPS[$i]}"
    NAME="${NAMES[$i]}"
    CMD="sbatch --nodelist=$NODE --job-name=p1-${NAME} $SBATCH_SCRIPT \"$MODELS\""

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $CMD"
    else
        JOB_ID=$(sbatch --parsable \
            --nodelist="$NODE" \
            --job-name="p1-${NAME}" \
            "$SBATCH_SCRIPT" "$MODELS")
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
