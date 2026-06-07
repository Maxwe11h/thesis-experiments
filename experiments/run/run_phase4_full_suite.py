"""Top-level driver for §5.4.6: launches every (algorithm, dim, instance-batch)
shard. Designed for parallel SLURM submission via `slurm/phase4_full_suite.sh`.

Usage:
  python -m experiments.run.run_phase4_full_suite --algorithm sage_winner --dim 10 \\
      --instance-start 0 --instance-end 100
"""
import argparse
from pathlib import Path

from experiments.phase4_full_suite_config import (
    ALGORITHMS,
    DIMS,
    RESULTS_DIR,
)
from experiments.phase4_full_suite_runner import run_shard


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', required=True, choices=list(ALGORITHMS))
    p.add_argument('--dim', type=int, required=True, choices=DIMS)
    p.add_argument('--instance-start', type=int, default=0)
    p.add_argument('--instance-end', type=int, default=1000)
    p.add_argument('--n-workers', type=int, default=1,
                   help='multiprocessing.Pool workers; 1 = sequential')
    args = p.parse_args()

    out_dir = Path(RESULTS_DIR) / args.algorithm / f'dim{args.dim}'
    instances = list(range(args.instance_start, args.instance_end))
    out = run_shard(args.algorithm, args.dim, instances, out_dir,
                    n_workers=args.n_workers)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
