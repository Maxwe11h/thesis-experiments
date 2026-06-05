#!/usr/bin/env python3
"""Re-classify Phase 1 failed candidates using the same BLADE compile-and-smoke
replay used for Phase 4. Caches results to analysis/figs_phase1/p1_failure_modes.csv.

Categories match analysis/phase4/failure_modes.py:
    code_generation | import_violation | interface_mismatch | runtime_error
plus 'unclassified_success' for replays whose smoke test now passes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.phase4.failure_modes import classify_failure  # noqa: E402

RESULTS_PHASE1 = REPO_ROOT / "results_phase1"
OUT_DIR = REPO_ROOT / "analysis" / "figs_phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "p1_failure_modes.csv"

MODELS = [
    "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "rnj-1-8b",
    "devstral-small-2-24b", "olmo3-7b", "olmo3-32b", "granite4-3b",
    "gemini-3-pro", "gemini-3-flash",
]
N_SEEDS = 5


def main() -> None:
    rows = []
    t0 = time.time()
    total_failed = 0

    for model in MODELS:
        model_failed = 0
        for seed in range(N_SEEDS):
            seed_dir = RESULTS_PHASE1 / model / f"seed-{seed}"
            run_dirs = sorted(seed_dir.glob("run-*"))
            if not run_dirs:
                continue
            log = run_dirs[0] / "log.jsonl"
            if not log.exists():
                continue

            with open(log) as fh:
                for i, line in enumerate(fh):
                    entry = json.loads(line)
                    fit = entry.get("fitness")
                    try:
                        f = float(fit)
                        failed = np.isinf(f) and f < 0
                    except (TypeError, ValueError):
                        failed = True
                    if not failed:
                        continue

                    code = entry.get("code", "") or ""
                    label, detail = classify_failure(code)
                    if label is None:
                        label = "unclassified_success"
                    rows.append({
                        "model": model, "seed": seed, "generation": i,
                        "name": entry.get("name", ""),
                        "label": label, "detail": detail,
                    })
                    model_failed += 1
                    total_failed += 1

        elapsed = time.time() - t0
        print(f"  {model}: {model_failed} failures classified  "
              f"(running total {total_failed}, {elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print("\nLabel breakdown:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
