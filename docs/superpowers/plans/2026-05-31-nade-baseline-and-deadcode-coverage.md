# NADE baseline + dead-code coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add van Stein's NeighborhoodAdaptiveDE (2025 MA-BBOB competition winner) as a second external baseline in the §5.4.6 full-suite comparison, and add a quantified dead-code/branch-coverage measurement of the four Stage-4 winners to §5.4.5.

**Architecture:** Task 1 reuses the existing full-suite harness unchanged — a harness-adapted NADE file is registered in `ALGORITHMS` and run as one more `(alg, dim, instance-batch)` shard, its parquet landing in the same `results_phase4_full_suite/` tree consumed by the existing figure/table export. Task 2 is a standalone local analysis script that drives each winner through the same `_run_once` evaluation under `coverage.py(branch=True)`, then AST-maps unreached lines.

**Tech Stack:** Python, NumPy 2.x, `ioh` + `iohblade` (BLADE), `coverage.py`, pandas/pyarrow, matplotlib; SLURM on rel-slurm; LaTeX (thesis).

**Conventions / preferences (must honor):**
- Do **not** modify existing experiment *config semantics*; Task 1 wiring is a purely additive key in `ALGORITHMS`.
- Commit messages must **not** include a `Co-Authored-By` line.
- Work on a feature branch. Do **not** push/merge or submit to rel-slurm without an explicit go from the user (committing locally per task is fine).
- ch6 future-work paragraph (`ch6_discussion.tex:79`) stays untouched.
- Run Python steps in the thesis env (locally: the conda/venv that already has `ioh`+`iohblade`; on server: `/local/$USER/conda_envs/thesis`). The NADE *unit* test needs only NumPy.

**Pre-flight (run once, not a task):**
```bash
cd /Users/maxharell/repos/thesis
git checkout -b nade-baseline-deadcode
mkdir -p baselines
```

---

## Task 1: NADE harness-adapted source file (+ provenance copy)

**Files:**
- Create: `baselines/neighborhood_adaptive_de_original.py` (verbatim provenance)
- Create: `baselines/neighborhood_adaptive_de.py` (harness-adapted)
- Test: `tests/test_neighborhood_adaptive_de.py`

- [ ] **Step 1: Save the provenance copy verbatim**

Copy the author-provided source unchanged (for reproducibility; not loaded by the harness):
```bash
cp /Users/maxharell/Downloads/NeighborhoodAdaptiveDE.py \
   baselines/neighborhood_adaptive_de_original.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_neighborhood_adaptive_de.py`:
```python
"""NADE harness-adapter contract tests (NumPy-only; no ioh needed).

Verifies the adapted NeighborhoodAdaptiveDE conforms to the full-suite harness
interface: cls(budget, dim) + __call__(func), draws from the global RNG (so the
harness per-run seed governs), respects the budget, and runs on NumPy 2.x.
"""
from pathlib import Path

import numpy as np

ADAPTED = Path(__file__).resolve().parents[1] / "baselines" / "neighborhood_adaptive_de.py"


def _load():
    src = ADAPTED.read_text()
    ns: dict = {}
    exec(compile(src, str(ADAPTED), "exec"), ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    return classes[-1]


def _counting_quadratic(budget):
    calls = [0]

    def f(x):
        calls[0] += 1
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    return f, calls


def test_runs_and_returns_finite_within_budget():
    NADE = _load()
    budget, dim = 600, 5
    f, calls = _counting_quadratic(budget)
    np.random.seed(0)
    algo = NADE(budget=budget, dim=dim)
    f_opt, x_opt = algo(f)
    assert np.isfinite(f_opt)
    assert x_opt is not None and len(x_opt) == dim
    # NADE evaluates whole generations; allow one extra generation of slack.
    assert calls[0] <= budget + algo.pop_size


def test_harness_seed_governs_so_seeds_differ():
    NADE = _load()

    def run(seed):
        f, _ = _counting_quadratic(600)
        np.random.seed(seed)
        return float(NADE(budget=600, dim=5)(f)[0])

    # No internal reseed -> different harness seeds give different runs.
    assert run(0) != run(1)


def test_no_np_inf_attribute_used():
    # np.Inf was removed in NumPy 2.0; importing/running must not touch it.
    src = ADAPTED.read_text()
    assert "np.Inf" not in src
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_neighborhood_adaptive_de.py -v`
Expected: FAIL (file `baselines/neighborhood_adaptive_de.py` does not exist yet).

- [ ] **Step 4: Write the adapted implementation**

Create `baselines/neighborhood_adaptive_de.py`. The DE body is byte-for-byte from the original; only the four documented plumbing changes are applied:
```python
# NeighborhoodAdaptiveDE -- van Stein (2025), GECCO Companion;
# winner of the 2025 MA-BBOB anytime competition (LLaMEA-SAGE ref [16]).
#
# Verbatim source kept in baselines/neighborhood_adaptive_de_original.py.
# This file changes ONLY the I/O plumbing so the algorithm runs under the
# §5.4.6 full-suite harness (experiments/phase4_full_suite_runner.py). The DE
# logic (neighbourhood mutation, orthogonal crossover, selection, adaptive
# F/CR, stagnation restart) is unchanged. Changes:
#   1. __init__ receives the harness's pre-multiplied budget (= 2000*dim)
#      instead of budget_factor; self.budget = budget.
#   2. Bounds hardcoded to the MA-BBOB box [-5, 5]; dim taken from the
#      constructor. The harness passes a plain FE-counting closure that has no
#      .bounds / .meta_data (the four LLM winners hardcode the box the same way).
#   3. __call__(self, func) draws randomness from the harness's per-run seed;
#      the original's internal np.random.seed(seed) is removed so the five
#      eval-seeds differ (otherwise NADE's seed variance would be exactly zero).
#   4. np.Inf -> np.inf (np.Inf was removed in NumPy 2.0).

import numpy as np


class NeighborhoodAdaptiveDE:
    def __init__(self, budget, dim=10, pop_size=50, adapt_freq=50,
                 stagnation_threshold=1000, learning_rate=0.1, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = pop_size
        self.adapt_freq = adapt_freq
        self.stagnation_threshold = stagnation_threshold
        self.learning_rate = learning_rate
        self.neighborhood_size = neighborhood_size
        self.f = 0.5
        self.cr = 0.9
        self.success_f = []
        self.success_cr = []
        self.best_fitness_history = []

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.evals = self.pop_size
        self.last_improvement = 0

        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx]
        self.best_fitness_history.append(self.f_opt)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                # Neighborhood-based Mutation
                neighbors = np.random.choice(np.arange(self.pop_size), self.neighborhood_size, replace=False)
                best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]

                idxs = np.random.choice(np.arange(self.pop_size), 2, replace=False)
                x1, x2 = self.population[idxs]

                v = self.population[best_neighbor_idx] + self.f * (x1 - x2)
                v = np.clip(v, self.lb, self.ub)

                # Orthogonal Crossover
                u = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_new = func(u)
                self.evals += 1

                # Selection
                if f_new < self.fitness[i]:
                    self.success_f.append(self.f)
                    self.success_cr.append(self.cr)
                    self.fitness[i] = f_new
                    self.population[i] = u

                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = u
                        self.best_fitness_history.append(self.f_opt)
                        self.last_improvement = self.evals

            # Adaptive Parameter Control
            if self.evals % self.adapt_freq == 0:
                if self.success_f:
                    self.f = (1 - self.learning_rate) * self.f + self.learning_rate * np.mean(self.success_f)
                    self.cr = (1 - self.learning_rate) * self.cr + self.learning_rate * np.mean(self.success_cr)
                self.f = np.clip(self.f, 0.1, 0.9)
                self.cr = np.clip(self.cr, 0.1, 1.0)
                self.success_f = []
                self.success_cr = []

            # Stagnation Check and Restart
            if self.evals - self.last_improvement > self.stagnation_threshold:
                self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.evals += self.pop_size
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                self.last_improvement = self.evals
                self.best_fitness_history.append(self.f_opt)

            if self.evals >= self.budget:
                break

        return self.f_opt, self.x_opt
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_neighborhood_adaptive_de.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add baselines/neighborhood_adaptive_de.py baselines/neighborhood_adaptive_de_original.py tests/test_neighborhood_adaptive_de.py
git commit -m "Add harness-adapted NeighborhoodAdaptiveDE baseline + provenance + tests"
```

---

## Task 2: Register NADE in the full-suite config (additive) and smoke-test it

**Files:**
- Modify: `experiments/phase4_full_suite_config.py:22-28` (`ALGORITHMS` dict — additive)

- [ ] **Step 1: Add the NADE entry**

In `experiments/phase4_full_suite_config.py`, change the `ALGORITHMS` dict from:
```python
ALGORITHMS = {
    'vanilla_winner':          'docs/stage4_winners/vanilla_winner.py',
    'neutral_winner':          'docs/stage4_winners/neutral_winner.py',
    'sage_winner':             'docs/stage4_winners/sage_winner.py',
    'combined_neutral_winner': 'docs/stage4_winners/combined_neutral_winner.py',
    'cma_es':                  'BUILTIN:cma_es',  # served by the runner directly
}
```
to (one added line; existing entries untouched):
```python
ALGORITHMS = {
    'vanilla_winner':          'docs/stage4_winners/vanilla_winner.py',
    'neutral_winner':          'docs/stage4_winners/neutral_winner.py',
    'sage_winner':             'docs/stage4_winners/sage_winner.py',
    'combined_neutral_winner': 'docs/stage4_winners/combined_neutral_winner.py',
    'cma_es':                  'BUILTIN:cma_es',  # served by the runner directly
    'neighborhood_adaptive_de': 'baselines/neighborhood_adaptive_de.py',  # external: 2025 MA-BBOB competition winner
}
```

- [ ] **Step 2: Local smoke run (in the thesis env)**

Run (small batch; CWD must be repo root so the relative path resolves):
```bash
python run_phase4_full_suite.py --algorithm neighborhood_adaptive_de \
  --dim 5 --instance-start 0 --instance-end 5
```
Expected: prints `wrote results_phase4_full_suite/neighborhood_adaptive_de/dim5/...parquet` with no traceback.

- [ ] **Step 3: Verify the parquet schema and seed variance**

Run:
```bash
python - <<'PY'
import pandas as pd, glob
p = sorted(glob.glob('results_phase4_full_suite/neighborhood_adaptive_de/dim5/*.parquet'))[-1]
df = pd.read_parquet(p)
assert set(df.columns) >= {'algorithm','dim','instance','eval_seed','aocc','curve'}, df.columns
print('rows', len(df), 'aocc range', round(df.aocc.min(),3), round(df.aocc.max(),3))
# instance 0 across the 5 seeds must NOT be identical (confirms harness seeding governs)
s = df[df.instance == 0].sort_values('eval_seed').aocc.tolist()
print('instance0 per-seed aocc', [round(x,4) for x in s])
assert len(set(round(x,6) for x in s)) > 1, 'seeds collapsed -- internal reseed not removed'
print('OK: seeds differ, schema matches')
PY
```
Expected: prints `OK: seeds differ, schema matches`; AOCC values are in a plausible (0,1) range.

- [ ] **Step 4: Remove the smoke artifacts (keep the tree clean for the real run)**

```bash
rm -rf results_phase4_full_suite/neighborhood_adaptive_de
```

- [ ] **Step 5: Commit**

```bash
git add experiments/phase4_full_suite_config.py
git commit -m "Register neighborhood_adaptive_de in full-suite ALGORITHMS (additive)"
```

---

## Task 3: Figure-script styling + bibliography entry for NADE

**Files:**
- Modify: `analysis/export_phase4_full_suite_figures.py:65-92` (ALGS + style dicts)
- Modify: `docs/thesisLatex/bibliography.bib` (add NADE citation)

- [ ] **Step 1: Add NADE to the figure-script algorithm constants**

In `analysis/export_phase4_full_suite_figures.py`, extend the four constants (append NADE; NADE is dashed like CMA-ES to flag it as an external baseline, with a distinct colour):
```python
ALGS = [
    "vanilla_winner",
    "neutral_winner",
    "sage_winner",
    "combined_neutral_winner",
    "cma_es",
    "neighborhood_adaptive_de",
]
ALG_LABELS = {
    "vanilla_winner": "Vanilla",
    "neutral_winner": "Neutral",
    "sage_winner": "SAGE",
    "combined_neutral_winner": "Combined",
    "cma_es": "CMA-ES",
    "neighborhood_adaptive_de": "NADE",
}
ALG_COLORS = {
    "vanilla_winner": "#888888",
    "neutral_winner": "#4e79a7",
    "sage_winner": "#E63946",
    "combined_neutral_winner": "#6A4C93",
    "cma_es": "#111111",
    "neighborhood_adaptive_de": "#2A9D8F",
}
ALG_LINESTYLES = {
    "vanilla_winner": "-",
    "neutral_winner": "-",
    "sage_winner": "-",
    "combined_neutral_winner": "-",
    "cma_es": "--",   # dashed to flag as external baseline
    "neighborhood_adaptive_de": "--",   # dashed: second external baseline
}
```

- [ ] **Step 2: Add the bibliography entry (only if absent)**

Run to check it does not already exist:
```bash
grep -n "vanStein2025NeighborhoodDE\|Neighborhood Adaptive Differential" docs/thesisLatex/bibliography.bib || echo "ABSENT -- add it"
```
If absent, append to `docs/thesisLatex/bibliography.bib`:
```bibtex
@inproceedings{vanStein2025NeighborhoodDE,
  author    = {van Stein, Niki},
  title     = {Neighborhood Adaptive Differential Evolution},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion)},
  year      = {2025},
  pages     = {1--2},
  publisher = {Association for Computing Machinery},
}
```

- [ ] **Step 3: Verify the figure script still imports/parses**

Run: `python -c "import ast; ast.parse(open('analysis/export_phase4_full_suite_figures.py').read()); print('parse OK')"`
Expected: `parse OK`. (Full figure regeneration is deferred to Task 5, after results exist.)

- [ ] **Step 4: Commit**

```bash
git add analysis/export_phase4_full_suite_figures.py docs/thesisLatex/bibliography.bib
git commit -m "Style NADE as second external baseline in full-suite figures; add bib entry"
```

---

## Task 4: SLURM script + launch on rel-slurm

**Files:**
- Create: `slurm/phase4_full_suite_nade.sh`

- [ ] **Step 1: Write the SLURM driver (mirrors `phase4_full_suite_optA.sh`, single algorithm)**

Create `slurm/phase4_full_suite_nade.sh`:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=p4_fs_nade
#SBATCH --output=/data/s3815129/slurm_logs/p4_fs_nade_%j.out
#SBATCH --error=/data/s3815129/slurm_logs/p4_fs_nade_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=L40s_students
#SBATCH --nodelist=saronite

# §5.4.6 full-suite: NeighborhoodAdaptiveDE only (2025 MA-BBOB competition winner).
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
  $ENV/bin/python run_phase4_full_suite.py \
    --algorithm "$ALG" --dim "$DIM" \
    --instance-start 0 --instance-end 1000 \
    --n-workers "$N_WORKERS"
  RC=$?
  T1=$(date +%s)
  echo "--- SHARD END   alg=$ALG dim=$DIM rc=$RC dt=$((T1 - T0))s ---"
done
echo "=== NADE SWEEP FINISHED $(date -Iseconds) ==="
```

- [ ] **Step 2: Verify the SLURM script is valid bash**

Run: `bash -n slurm/phase4_full_suite_nade.sh && echo "syntax OK"`
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add slurm/phase4_full_suite_nade.sh
git commit -m "Add SLURM driver for NADE full-suite sweep"
```

- [ ] **Step 4: Launch on rel-slurm — STOP and get user go-ahead first**

This step pushes code to the server and submits a job; per the user's preference, **ask before doing it.** Once approved, get the new files onto the server (e.g. `git push` then `git pull` in `$HOME/thesis` on the server, or `rsync` the working tree), then submit via the `/rel-connect` skill / ssh:
```
sbatch slurm/phase4_full_suite_nade.sh
```
Monitor with `squeue -u $USER` and the `*_nade_*.out` log. On completion, copy results back so the local tree has `results_phase4_full_suite/neighborhood_adaptive_de/dim{5,10,20}/*.parquet` (the existing full-suite results live under `/data/$USER/results_phase4_full_suite`; mirror the new alg dir back to the repo's `results_phase4_full_suite/`). Note: the four winners' shards are unaffected — this only adds the `neighborhood_adaptive_de/` subtree.

---

## Task 5: Update §5.4.6 (table, prose, figure caption) from results — AFTER the run

**Files:**
- Modify: `docs/thesisLatex/chapters/ch5_results.tex:435,437-453,458,462`

- [ ] **Step 1: Regenerate the full-suite figures + summary CSV**

Run (thesis env, repo root, with NADE results present locally):
```bash
python analysis/export_phase4_full_suite_figures.py
cat analysis/figs_phase4_full_suite/p46_final_aocc_summary.csv
```
Expected: the EAF figure now includes a dashed NADE curve; the CSV has a `NADE` row with mean AOCC per dim. Record NADE's three means (call them `N5, N10, N20`) and re-read every algorithm's per-dim means (winners may need bolding recomputed).

- [ ] **Step 2: Add the NADE row to `tab:phase4-fullsuite-summary`**

In `ch5_results.tex`, insert a NADE row after the CMA-ES baseline row (line 450). Use the values from the CSV; keep the existing four winner rows and CMA-ES row as-is unless the per-dim bold (best entry per column) changes — recompute bold across all six rows:
```latex
baseline  & CMA-ES~\cite{hansen2001cmaes} & 0.728 & 0.701 & 0.650 \\
baseline  & \texttt{NeighborhoodAdaptiveDE}~\cite{vanStein2025NeighborhoodDE} & N5 & N10 & N20 \\
```
(Replace `N5/N10/N20` with the CSV numbers to three decimals. If a NADE cell is the column max, bold it with `\textbf{...}` and unbold whichever winner previously held it.)

- [ ] **Step 3: Update the §5.4.6 setup prose (line 435)**

Replace the sentence beginning "To position the four winners…" so it names both external baselines, says **six** algorithms, and updates the compute total (6 × 3.5e9... = per-alg FEs `1000*5*2000*(5+10+20)=3.5e8`, ×6 ≈ `2.1e9`):
```latex
To position the four winners against a broader benchmark, the best-found algorithm from each condition (\cref{tab:phase4-winners}) is re-evaluated against two external baselines with no LLM in the loop: a CMA-ES~\cite{hansen2001cmaes} reference and \texttt{NeighborhoodAdaptiveDE}~\cite{vanStein2025NeighborhoodDE}, the winner of the 2025 MA-BBOB competition. Each of the six algorithms runs on all $1{,}000$ MA-BBOB functions across 5 instance seeds and three dimensionalities $d \in \{5, 10, 20\}$, under a $2{,}000d$ FE budget per run. The total compute is approximately $2.1 \times 10^9$ FEs across the six algorithms.
```

- [ ] **Step 4: Update the EAF figure caption (line 458)**

Change the baseline sentence from a single dashed CMA-ES line to two external baselines, and adjust the final claim per the actual curves:
```latex
The two dashed lines mark the external baselines (CMA-ES and NeighborhoodAdaptiveDE).
```
Then rewrite the closing "sit above it by a stable margin" clause to match what the regenerated EAF shows relative to NADE (e.g. whether the LLM variants sit above/below NADE).

- [ ] **Step 5: Update the results-reading paragraph (line 462) per the result rule**

Rewrite the comparative sentences using this deterministic rule from the CSV: for each dimension `d`, compare each winner's mean to `N{d}` (NADE) and to the CMA-ES mean, and state the ordering factually (e.g. "All four LLM variants beat CMA-ES at every `d`; against NADE they lead at `d=…` and trail at `d=…`"). Do not assert a margin the figure does not show.

- [ ] **Step 6: Build the thesis to confirm no LaTeX/citation errors**

Run (repo root):
```bash
cd docs/thesisLatex && latexmk -pdf -interaction=nonstopmode thesis.tex >/tmp/latexmk.log 2>&1; tail -5 /tmp/latexmk.log; cd -
```
Expected: build completes; `grep -i "undefined" /tmp/latexmk.log` shows no undefined `vanStein2025NeighborhoodDE` citation.

- [ ] **Step 7: Commit**

```bash
git add docs/thesisLatex/chapters/ch5_results.tex analysis/figs_phase4_full_suite docs/thesisLatex/figures
git commit -m "Report NADE second external baseline in §5.4.6 (table, EAF, prose)"
```

---

## Task 6: Coverage core — class loader + AST mapping (pure, no coverage/ioh)

**Files:**
- Create: `analysis/winner_coverage.py` (start the module: loader + AST mapper)
- Create: `tests/analysis/fixtures/winner_with_deadcode.py` (synthetic winner)
- Test: `tests/analysis/test_winner_coverage.py`

- [ ] **Step 1: Add the synthetic winner fixture with known dead code**

Create `tests/analysis/fixtures/winner_with_deadcode.py` (a minimal CMA-like shell with a never-taken bare `except`, an unreachable helper, and a live path):
```python
import numpy as np


class FixtureWinner:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0

    def _never_called_helper(self):       # dead: never invoked
        return np.zeros(self.dim)

    def __call__(self, func):
        best = np.inf
        x = np.random.uniform(self.lb, self.ub, self.dim)
        for _ in range(self.budget):
            cand = np.clip(x + 0.1 * np.random.randn(self.dim), self.lb, self.ub)
            try:
                y = float(func(cand))
            except Exception:
                y = 1e30                  # dead: func never raises here
            if y < best:
                best = y
                x = cand
        return best, x
```

- [ ] **Step 2: Write the failing test for loader + AST mapper**

Create `tests/analysis/test_winner_coverage.py`:
```python
from pathlib import Path

import numpy as np

from analysis.winner_coverage import (
    load_traced_class,
    map_missing_to_ast,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "winner_with_deadcode.py"


def test_load_traced_class_returns_last_class_with_real_filename():
    cls = load_traced_class(FIXTURE)
    assert cls.__name__ == "FixtureWinner"
    # __call__'s code object must carry the real file path (needed for coverage).
    assert cls.__call__.__code__.co_filename == str(FIXTURE)


def test_map_missing_to_ast_labels_constructs():
    src = FIXTURE.read_text()
    # Lines of the dead helper body and the except body (1-indexed) from the fixture.
    lines = src.splitlines()
    helper_line = next(i + 1 for i, l in enumerate(lines) if "return np.zeros(self.dim)" in l)
    except_body = next(i + 1 for i, l in enumerate(lines) if "y = 1e30" in l)
    constructs = map_missing_to_ast(FIXTURE, [helper_line, except_body])
    kinds = {c["line"]: c["construct"] for c in constructs}
    assert "FunctionDef:_never_called_helper" in kinds[helper_line]
    assert "ExceptHandler" in kinds[except_body]
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/analysis/test_winner_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError: analysis.winner_coverage` / import error.

- [ ] **Step 4: Implement the loader + AST mapper**

Create `analysis/winner_coverage.py`:
```python
#!/usr/bin/env python3
"""Dead-code / branch-coverage measurement for the four Stage-4 winners (§5.4.5).

Drives each winner through the full-suite evaluation harness under coverage.py
(branch=True), then maps every unreached line back to its AST construct. Reports
per winner: % of statements/branches executed, never-executed constructs, and
how often each bare `except` handler actually fired.

Run locally (thesis env):  python analysis/winner_coverage.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
WINNERS = {
    "vanilla":  REPO_ROOT / "docs/stage4_winners/vanilla_winner.py",
    "neutral":  REPO_ROOT / "docs/stage4_winners/neutral_winner.py",
    "sage":     REPO_ROOT / "docs/stage4_winners/sage_winner.py",
    "combined": REPO_ROOT / "docs/stage4_winners/combined_neutral_winner.py",
}


def load_traced_class(winner_path: Path) -> type:
    """Compile+exec the winner with its REAL filename so coverage can attribute
    executed lines to it, returning the last-defined class (harness convention)."""
    src = Path(winner_path).read_text()
    code = compile(src, str(winner_path), "exec")
    ns: dict = {}
    exec(code, ns, ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    if not classes:
        raise RuntimeError(f"no class defined in {winner_path}")
    return classes[-1]


def _enclosing_label(tree: ast.AST, line: int) -> str:
    """Best (innermost) AST construct label covering `line`."""
    best = ("Module", -1, 1 << 30)  # (label, depth, span)
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        if start <= line <= end:
            span = end - start
            if isinstance(node, ast.FunctionDef):
                label = f"FunctionDef:{node.name}"
            elif isinstance(node, ast.ExceptHandler):
                label = "ExceptHandler"
            elif isinstance(node, ast.If):
                label = "If"
            elif isinstance(node, ast.While):
                label = "While"
            elif isinstance(node, ast.For):
                label = "For"
            else:
                continue
            # prefer the tightest (smallest-span) matching construct
            if span < best[2]:
                best = (label, 0, span)
    return best[0]


def map_missing_to_ast(winner_path: Path, missing_lines: list[int]) -> list[dict]:
    """For each unreached line, attach the innermost interesting AST construct."""
    tree = ast.parse(Path(winner_path).read_text())
    out = []
    for ln in sorted(set(missing_lines)):
        out.append({"line": ln, "construct": _enclosing_label(tree, ln)})
    return out
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/analysis/test_winner_coverage.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add analysis/winner_coverage.py tests/analysis/test_winner_coverage.py tests/analysis/fixtures/winner_with_deadcode.py
git commit -m "Add winner-coverage core: traced loader + AST construct mapping"
```

---

## Task 7: Coverage measurement under coverage.py (branch=True)

**Files:**
- Modify: `analysis/winner_coverage.py` (add `measure_coverage`)
- Modify: `tests/analysis/test_winner_coverage.py` (add measurement test)

- [ ] **Step 1: Ensure coverage.py is installed (thesis env)**

Run: `python -c "import coverage" 2>/dev/null || pip install coverage`
Then: `python -c "import coverage; print(coverage.__version__)"`
Expected: a version string (>= 7).

- [ ] **Step 2: Write the failing measurement test**

Append to `tests/analysis/test_winner_coverage.py`:
```python
def test_measure_coverage_flags_dead_lines_and_except():
    from analysis.winner_coverage import measure_coverage

    def driver(cls):
        # Exercise only the live path on a trivial quadratic; func never raises.
        np.random.seed(0)
        algo = cls(budget=200, dim=4)
        algo(lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)))

    res = measure_coverage(FIXTURE, driver)
    assert 0.0 < res["pct_lines"] < 100.0          # some code is dead
    # The dead helper and the never-taken except must be reported unreached.
    dead_constructs = {c["construct"] for c in res["dead_constructs"]}
    assert any("FunctionDef:_never_called_helper" in c for c in dead_constructs)
    assert "ExceptHandler" in dead_constructs
    # The bare except never fired.
    assert res["except_handlers_total"] >= 1
    assert res["except_handlers_triggered"] == 0
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/analysis/test_winner_coverage.py::test_measure_coverage_flags_dead_lines_and_except -v`
Expected: FAIL (`measure_coverage` not defined).

- [ ] **Step 4: Implement `measure_coverage`**

Append to `analysis/winner_coverage.py`:
```python
def _except_handler_lines(winner_path: Path) -> list[list[int]]:
    """Return, per bare/except handler, the set of body line numbers."""
    tree = ast.parse(Path(winner_path).read_text())
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body_lines = []
            for stmt in node.body:
                body_lines.extend(
                    range(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno) + 1)
                )
            handlers.append(sorted(set(body_lines)))
    return handlers


def measure_coverage(winner_path: Path, driver: Callable[[type], None]) -> dict:
    """Run `driver(loaded_class)` under coverage.py(branch=True) and summarise."""
    import coverage

    winner_path = Path(winner_path)
    cov = coverage.Coverage(branch=True, source=[str(winner_path)])
    cls = load_traced_class(winner_path)  # compiled with real filename
    cov.start()
    try:
        driver(cls)
    finally:
        cov.stop()

    # analysis2 -> (filename, statements, excluded, missing, missing_formatted)
    _, statements, _excluded, missing, _ = cov.analysis2(str(winner_path))
    n_stmt = len(statements)
    n_missing = len(missing)
    pct_lines = 100.0 * (n_stmt - n_missing) / n_stmt if n_stmt else 100.0

    # Branch totals from the JSON report summary (public, stable API).
    import json, tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cov.json_report(outfile=tmp)
        data = json.loads(Path(tmp).read_text())
    finally:
        os.unlink(tmp)
    fkey = next((k for k in data["files"] if Path(k).name == winner_path.name), None)
    summ = data["files"][fkey]["summary"] if fkey else {}
    n_branches = summ.get("num_branches", 0)
    covered_branches = summ.get("covered_branches", 0)
    pct_branches = (100.0 * covered_branches / n_branches) if n_branches else 100.0

    # Which except handlers never fired (no body line executed).
    handlers = _except_handler_lines(winner_path)
    missing_set = set(missing)
    triggered = sum(1 for body in handlers if any(ln not in missing_set for ln in body))

    return {
        "winner": winner_path.stem,
        "n_statements": n_stmt,
        "n_missing": n_missing,
        "pct_lines": round(pct_lines, 1),
        "n_branches": n_branches,
        "pct_branches": round(pct_branches, 1),
        "dead_lines": sorted(missing),
        "dead_constructs": map_missing_to_ast(winner_path, list(missing)),
        "except_handlers_total": len(handlers),
        "except_handlers_triggered": triggered,
    }
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/analysis/test_winner_coverage.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add analysis/winner_coverage.py tests/analysis/test_winner_coverage.py
git commit -m "Add coverage.py-based measurement (line/branch + except-trigger tally)"
```

---

## Task 8: Production driver over MA-BBOB + CSV output; run locally

**Files:**
- Modify: `analysis/winner_coverage.py` (add MA-BBOB driver + `main`)
- Create: `analysis/winner_coverage_results.csv` (generated output)

- [ ] **Step 1: Add the MA-BBOB driver and `main()`**

Append to `analysis/winner_coverage.py`:
```python
def mabbob_driver(dims=(5, 10, 20), n_instances=40, n_seeds=3) -> Callable[[type], None]:
    """Driver that runs a winner over a MA-BBOB sample at the full 2000*d budget,
    reusing the §5.4.6 runner so coverage reflects real evaluation behaviour."""
    from experiments.phase4_full_suite_runner import _run_once

    def driver(cls):
        factory = lambda budget, dim: cls(budget=budget, dim=dim)
        for dim in dims:
            for inst in range(n_instances):
                for seed in range(n_seeds):
                    _run_once(factory, dim, inst, seed)

    return driver


def main() -> None:
    import csv
    driver = mabbob_driver()
    rows = []
    for name, path in WINNERS.items():
        res = measure_coverage(path, driver)
        res["condition"] = name
        rows.append(res)
        print(f"{name:9s} lines {res['pct_lines']:5.1f}%  "
              f"branches {res['pct_branches']:5.1f}%  "
              f"dead_lines {res['n_missing']:3d}  "
              f"except {res['except_handlers_triggered']}/{res['except_handlers_total']} fired")
        for c in res["dead_constructs"]:
            print(f"    L{c['line']:>3}  {c['construct']}")

    out = REPO_ROOT / "analysis" / "winner_coverage_results.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "winner", "n_statements", "n_missing",
                    "pct_lines", "n_branches", "pct_branches",
                    "except_handlers_total", "except_handlers_triggered"])
        for r in rows:
            w.writerow([r["condition"], r["winner"], r["n_statements"], r["n_missing"],
                        r["pct_lines"], r["n_branches"], r["pct_branches"],
                        r["except_handlers_total"], r["except_handlers_triggered"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the real measurement (thesis env, repo root)**

Run: `python analysis/winner_coverage.py`
Expected: a per-winner summary line for vanilla/neutral/sage/combined, the listed dead constructs, and `wrote analysis/winner_coverage_results.csv`. Sanity: expect ~3 of 4 winners to show `except 0/1` (bare except never fired), matching the §5.4.5 claim.

- [ ] **Step 3: Robustness check — dead set is stable, not under-sampled**

Run a larger sample and confirm the dead-line counts do not shrink (i.e. the unreached code is genuinely inert, not just rarely hit):
```bash
python - <<'PY'
from pathlib import Path
from analysis.winner_coverage import measure_coverage, mabbob_driver, WINNERS
small = mabbob_driver(n_instances=40, n_seeds=3)
big   = mabbob_driver(n_instances=80, n_seeds=5)
for name, path in WINNERS.items():
    a = measure_coverage(path, small)["n_missing"]
    b = measure_coverage(path, big)["n_missing"]
    print(f"{name:9s} dead_lines small={a} big={b}  {'STABLE' if b<=a else 'SHRANK?'}")
PY
```
Expected: `big <= small` for every winner (more sampling never *increases* coverage of truly-dead code; equality = saturated).

- [ ] **Step 4: Commit**

```bash
git add analysis/winner_coverage.py analysis/winner_coverage_results.csv
git commit -m "Add MA-BBOB coverage driver; generate winner dead-code results"
```

---

## Task 9: Report the coverage measurement in §5.4.5

**Files:**
- Modify: `docs/thesisLatex/chapters/ch5_results.tex:428` (the "functionally inert" paragraph)

- [ ] **Step 1: Read the generated numbers**

Run: `column -s, -t analysis/winner_coverage_results.csv`
Note, per winner: `pct_lines`, `n_missing`, and `except_handlers_triggered/except_handlers_total`.

- [ ] **Step 2: Insert a quantification table + sentences after line 428**

Add (using the CSV numbers; this *augments* the existing qualitative paragraph rather than replacing it — the unused-but-executed observations like `success_history` stay as static facts, since coverage only measures *execution*):
```latex
To move these observations beyond illustration, each winner was run under branch-level coverage tracing on a sample of MA-BBOB instances at the full $2{,}000d$ budget across $d \in \{5,10,20\}$; \cref{tab:winner-coverage} reports the fraction of statements actually executed and the fate of the bare \texttt{except} handlers. Across the four winners, <PCT_RANGE>\% of statements are reached, and in <K> of the four the bare \texttt{except} handler that guards the eigendecomposition never executes a single time over the entire sample, confirming that the identity-covariance fallback is dead under normal evaluation rather than a live safeguard. The remaining inert patches (the unused \texttt{success\_history} list, the misnamed comments) are not unexecuted code but executed code with no downstream effect, and are reported here as static observations rather than coverage results.
```
Then add the table:
```latex
\begin{table}[t]
\centering
\caption{Statement coverage of the four Stage-4 winners under MA-BBOB evaluation (full $2{,}000d$ budget, $d \in \{5,10,20\}$). ``except fired'' counts how many of the algorithm's bare \texttt{except} handlers executed at least once.}
\label{tab:winner-coverage}
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Cond.} & \textbf{Algorithm} & \textbf{\% stmts exec.} & \textbf{dead lines} & \textbf{except fired} \\
\midrule
vanilla  & \texttt{IGA\_CMA}          & V1 & V2 & V3 \\
neutral  & \texttt{BIPOP\_MA\_CMA\_ES} & N1 & N2 & N3 \\
sage     & \texttt{CMA\_EES}          & S1 & S2 & S3 \\
combined & \texttt{OM\_CMA\_EIS}       & C1 & C2 & C3 \\
\bottomrule
\end{tabular}
\end{table}
```
Replace `V1..C3`, `<PCT_RANGE>`, and `<K>` with the CSV values (`except fired` shown as `t/T`). These are filled from data, not left as placeholders.

- [ ] **Step 3: Build the thesis to confirm it compiles**

Run: `cd docs/thesisLatex && latexmk -pdf -interaction=nonstopmode thesis.tex >/tmp/latexmk2.log 2>&1; tail -5 /tmp/latexmk2.log; cd -`
Expected: build completes; `tab:winner-coverage` resolves with no undefined reference.

- [ ] **Step 4: Commit**

```bash
git add docs/thesisLatex/chapters/ch5_results.tex
git commit -m "Quantify §5.4.5 dead-code claims with coverage measurement"
```

---

## Self-review notes (coverage of the spec)

- Spec Task 1 (NADE baseline): adapter file (T1), additive config (T2), figure styling + bib (T3), SLURM + launch (T4), §5.4.6 write-up (T5). ✓
- Spec Task 2 (dead-code coverage, "Both"): line+branch via coverage.py (T7) + AST mapping (T6), MA-BBOB driver + CSV (T8), §5.4.5 write-up incl. dead-vs-rare robustness check (T8 step 3) and honest framing of unused-but-executed code (T9). ✓
- NADE seeding (drop internal reseed) verified by `test_harness_seed_governs` (T1) and the smoke seed-variance assert (T2). ✓
- `np.Inf`→`np.inf` enforced by `test_no_np_inf_attribute_used` (T1). ✓
- ch6:79 untouched: not referenced by any task. ✓
- Result-dependent LaTeX (T5/T9) uses deterministic fill-from-CSV procedures, not placeholders. ✓
