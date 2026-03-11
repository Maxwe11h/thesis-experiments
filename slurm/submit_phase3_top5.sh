#!/bin/bash
# Submit all 15 Phase 3 top-5 conditions as separate SLURM jobs.
# Each job runs one condition (5 seeds sequentially), no GPU needed.
#
# Usage:
#   cd ~/thesis
#   bash slurm/submit_phase3_top5.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/slurm

FEATURES=(
    avg_improvement
    intensification_ratio
    fitness_plateau_fraction
    step_size_autocorrelation
    improvement_spatial_correlation
)
FORMATS=(neutral directional comparative)

count=0
for feat in "${FEATURES[@]}"; do
    for fmt in "${FORMATS[@]}"; do
        cond="${fmt}-${feat}"

        # Check condition exists
        python -c "from experiments.phase3_config import get_conditions; assert '${cond}' in get_conditions()" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "SKIP $cond (not in condition registry)"
            continue
        fi

        echo "Submitting: $cond"
        sbatch --job-name="p3-${cond}" slurm/phase3.sbatch "$cond"
        ((count++))
        sleep 1
    done
done

echo ""
echo "Submitted $count jobs. Monitor with: squeue -u \$USER"
