# Worker Pool Benchmark Results

**Date**: 2026-02-23
**Machine**: macOS (Darwin 25.2.0)
**Algorithm**: RandomSearch (known-good baseline)
**Config**: 3 training instances [0, 7, 14], dim=5, budget_factor=2000
**N evals**: 5 per mode

## Raw Results

### Subprocess-per-eval mode

| Eval | Time (s) | Fitness |
|------|----------|---------|
| 1    | 90.51    | 0.2910  |
| 2    | 3.42     | 0.2910  |
| 3    | 3.17     | 0.2910  |
| 4    | 3.15     | 0.2910  |
| 5    | 3.18     | 0.2910  |

Eval 1 includes one-time venv creation (~87s).

### Worker pool mode

| Eval | Time (s) | Fitness |
|------|----------|---------|
| 1    | 87.84    | 0.2910  |
| 2    | 1.05     | 0.2910  |
| 3    | 1.02     | 0.2910  |
| 4    | 1.03     | 0.2910  |
| 5    | 1.04     | 0.2910  |

Eval 1 includes one-time venv creation (~87s).

## Summary

| Metric                          | Subprocess | Worker Pool |
|---------------------------------|------------|-------------|
| Steady-state per-eval (mean)    | ~3.2s      | ~1.0s       |
| One-time venv creation          | ~88s       | ~88s        |
| Mean (all 5 evals)              | 20.69s     | 18.40s      |
| Std (all 5 evals)               | 34.91s     | 34.72s      |

**Steady-state speedup: ~3.1x**

## Projected Savings (100-eval experiment)

- Subprocess overhead: 100 × 3.2s = ~320s
- Worker pool overhead: 100 × 1.0s = ~100s
- **Savings: ~220s (3.7 min) per experiment run**
- Across 3 seeds: ~11 min saved
- Across all 4 conditions × 3 seeds: ~44 min saved

## Notes

- Both modes produce identical fitness values (0.2910), confirming correctness.
- The worker pool keeps a persistent subprocess alive, avoiding the ~2.2s
  per-eval cost of spawning a new Python process, pickling/unpickling the
  problem, and re-importing modules each time.
- Worker is recycled every 50 evaluations (configurable) to prevent memory leaks.
- `llamea/utils.py` is imported directly via `importlib.util.spec_from_file_location`
  to avoid pulling in the full LLaMEA package (which requires `lizard`, `networkx`,
  etc. that aren't needed for evaluation).
