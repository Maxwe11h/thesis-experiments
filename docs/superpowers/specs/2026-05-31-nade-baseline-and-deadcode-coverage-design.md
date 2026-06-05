# Design: NADE external baseline + quantified dead-code coverage of Stage-4 winners

Date: 2026-05-31
Status: approved (pending spec review)

## Motivation

Two independent thesis additions, both responding to examiner/reviewer feedback on Chapter 5.

1. **NeighborhoodAdaptiveDE (NADE) baseline.** §5.4.6 ("Full-Suite Generalisation")
   currently compares the four LLM-generated winners against a single external
   baseline (hand-tuned CMA-ES) on all 1 000 MA-BBOB instances. We add a second,
   stronger external baseline: van Stein's GECCO-2025 MA-BBOB **competition
   winner** NeighborhoodAdaptiveDE (LLaMEA-SAGE ref [16]; the "Competition winner
   2025" line in that paper's Fig. 11). This positions the winners against a
   human-competitive, MA-BBOB-tuned reference rather than only a generic CMA-ES.

2. **Quantified dead-code measurement.** §5.4.5 claims several winner code blocks
   are "functionally inert" and that "bare `except` clauses in three of the four"
   silently fall back to identity covariance (`ch5_results.tex:428`). Reviewer
   feedback: these are anecdotes unless we measure what fraction of the code is
   actually reached during a run. We add a branch-coverage + dead-code measurement
   over the four winners and report it **in the results section**.

   Out of scope: the ch6 future-work paragraph (`ch6_discussion.tex:79`) proposes a
   *different, larger* idea — AST-diffing + execution-tracing used to **gate
   mutations inside the LLaMEA loop**. That paragraph is intentionally left
   unchanged. This work only adds a static post-hoc measurement to §5.4.5.

## Locked decisions

- **Task 1 wiring:** additive edit — add one key to `ALGORITHMS` in
  `experiments/phase4_full_suite_config.py`; reuse the existing runner/driver
  unchanged. Purely additive; the existing 5-algorithm results are untouched.
- **Task 2 runtime:** run the coverage analysis locally (needs `ioh` + BLADE,
  both present locally; a few hundred short runs).
- **Task 2 metric:** *Both* — `coverage.py` (branch=True) line+branch numbers as
  the headline, plus mapping of each unreached line/branch back to its AST
  construct for the prose.
- **NADE seeding:** drop the algorithm's internal `np.random.seed(0)` so it draws
  its randomness from the harness's per-run seed, exactly like the other five
  algorithms. Otherwise its five eval-seeds collapse to identical runs and its
  boxplot variance is artificially zero. DE logic is otherwise byte-for-byte.
- **NADE hyperparameters:** the provided file's `__init__` defaults
  (`pop_size=50, adapt_freq=50, stagnation_threshold=1000, learning_rate=0.1,
  neighborhood_size=5`, initial `f=0.5`, `cr=0.9`) are taken as the competition
  config.

---

## Task 1 — NADE full-suite baseline

### Harness recap (existing, unchanged)
`run_phase4_full_suite.py --algorithm A --dim D` → `run_shard` → `_run_once`:
builds an `ioh.problem.ManyAffine` instance, wraps it in a plain `wrapped(x)`
closure that counts FEs and records the best-so-far `curve`, instantiates the
algorithm via `cls(budget=2000*D, dim=D)`, calls `algo(wrapped)`, computes AOCC
from the curve, and writes one parquet per (alg, dim, instance-batch) under
`results_phase4_full_suite/<alg>/dim<d>/`. `_load_user_algo` execs the algorithm
file and takes the **last-defined class**.

### Files

**NEW `baselines/neighborhood_adaptive_de_original.py`** — the verbatim source as
provided (`/Users/maxharell/Downloads/NeighborhoodAdaptiveDE.py`), kept for
provenance. Not loaded by the harness.

**NEW `baselines/neighborhood_adaptive_de.py`** — harness-adapted NADE. The DE
body (neighbourhood mutation, orthogonal crossover, selection, adaptive F/CR,
stagnation restart) is copied **verbatim**. Only the following I/O plumbing
changes are made, each justified:

| # | Change | Reason | Fidelity |
|---|--------|--------|----------|
| 1 | `__init__(self, budget, dim, pop_size=50, adapt_freq=50, stagnation_threshold=1000, learning_rate=0.1, neighborhood_size=5)`; set `self.budget = budget` | harness passes pre-multiplied `budget = 2000*dim`, not `budget_factor` | identical value (2000·dim) |
| 2 | hardcode `self.lb=-5.0, self.ub=5.0`; replace `func.bounds.lb/ub` → `self.lb/ub`, `func.meta_data.n_variables` → `self.dim` | harness passes a plain `wrapped` closure with no `.bounds`/`.meta_data`; matches how the 4 winners already work | MA-BBOB bounds are exactly [-5,5]; dim from constructor equals problem dim |
| 3 | `__call__(self, func)` — drop the `seed` param and internal `np.random.seed` | rely on harness per-run seed (locked decision) | restores genuine seed-to-seed variance; comparable to other algorithms |
| 4 | `np.Inf` → `np.inf` | `np.Inf` removed in NumPy ≥2.0 → would crash on line 1 of `__call__` and silently score ~0 | bugfix; no behavioural change on NumPy <2.0 |

A header comment in the file lists changes 1–4 explicitly for reproducibility.
The class remains named `NeighborhoodAdaptiveDE` and is the only/last class in
the file.

**EDIT `experiments/phase4_full_suite_config.py`** — add one entry:
```python
'neighborhood_adaptive_de': 'baselines/neighborhood_adaptive_de.py',
```
(additive; existing keys untouched).

**EDIT `analysis/export_phase4_full_suite_figures.py`** — append
`"neighborhood_adaptive_de"` to `ALGS`; add label `"NADE"`, a distinct colour,
and a dashed linestyle (matching `cma_es`, flagging it as an external baseline).
Results are auto-discovered by the per-alg `glob`, so no other plot change is
needed. (Analysis script, not an experiment config — editing is fine.)

**NEW `slurm/phase4_full_suite_nade.sh`** — mirrors `phase4_full_suite_optA.sh`
but loops only `ALG=neighborhood_adaptive_de` over `DIMS=(5 10 20)`,
`--instance-start 0 --instance-end 1000`, `--n-workers $SLURM_CPUS_PER_TASK`,
writing to `$PHASE4_FULL_SUITE_DIR`. CPU-only.

**EDIT `docs/thesisLatex/chapters/ch5_results.tex`** §5.4.6 — add a NADE row to
`tab:phase4-fullsuite-summary`, add its EAF curve mention/caption tweak, and
update the prose. Data is inserted first; the comparative conclusion (does the
winner beat NADE?) is written only after results land.

### Data flow
Identical to existing algorithms; NADE parquet lands in the same `RESULTS_DIR`
tree and is consumed by the existing figure/table export.

### Compute
1 alg × 3 dims × 1 000 instances × 5 seeds = 15 000 runs at budget 2000·d
(d=20 dominates). By the existing 5-alg projection (~24 CPU-h / 32 CPUs ≈ 45 min),
~5 CPU-h ≈ ~10 min wall on 32 CPUs; allow headroom (DE may be slower per FE).

### Validation
- Local smoke: `run_phase4_full_suite.py --algorithm neighborhood_adaptive_de
  --dim 5 --instance-start 0 --instance-end 5`. Confirm: parquet schema matches
  (`algorithm,dim,instance,eval_seed,aocc,curve`); AOCC in a plausible range; the
  5 eval-seeds differ (seed-variance > 0, confirming change #3).
- Then submit `slurm/phase4_full_suite_nade.sh` on rel-slurm.

---

## Task 2 — Dead-code / branch coverage of the four winners

### Files

**NEW `analysis/winner_coverage.py`** — measurement script.

- For each winner file in `docs/stage4_winners/{vanilla,neutral,sage,combined_neutral}_winner.py`:
  - Drive it through the same evaluation path used by the full-suite runner
    (reuse `experiments/phase4_full_suite_runner._run_once` /
    `_ensure_ma_bbob_data` / `_load_user_algo`), but execute the winner's source
    via `compile(src, winner_path, 'exec')` under a
    `coverage.Coverage(branch=True, source=[winner_path])` session so coverage
    attributes lines/branches to the real file.
  - Sample budget: all three dims `d ∈ {5,10,20}` × ~30–50 MA-BBOB instances ×
    a few seeds, at the full 2000·d budget, so genuinely-live-but-conditional
    branches (stagnation restart, boundary handling) get a fair chance to fire.
    Only code unreached across **all** runs is reported as inert.
- Collect from coverage's JSON/analysis API: executable statements, missing
  statements, % lines covered, branch count, % branches taken, and the list of
  partial/never-taken branches.
- AST mapping: parse each winner with `ast`; for every missing line / never-taken
  branch, locate the enclosing node (`FunctionDef`, `ExceptHandler`, `If`,
  `While`) and emit a human-readable label, e.g. *"`except` fallback to identity
  covariance: 0 / N evaluations triggered it"*, *"helper `_x` defined, never
  called"*, *"`success_history` appended, never read"*.
- Specifically tally, per winner, how many times each bare-`except` handler body
  executed (expected 0) to substantiate the "silently fall back … on numerical
  failure" claim, and confirm the 3-of-4 count.

### Outputs
- `analysis/winner_coverage_results.csv` — one row per winner: statements,
  % executed, branches, % taken, # dead lines, # bare-except handlers never
  triggered.
- Optional small LaTeX table `tab:winner-coverage`.
- 2–3 quantified sentences added to the `ch5_results.tex:428` paragraph,
  converting the qualitative claims into measured numbers with an honest
  "unreached under evaluation" caveat (dead vs. merely-rare).

### Validation
- Re-run with two different sample sizes; live-code coverage should saturate
  (stable %), and the dead set should not shrink with more samples — evidence the
  unreached code is genuinely inert rather than under-sampled.

---

## Error handling
- The runner already wraps `algo(wrapped)` in try/except and scores whatever curve
  exists; the NADE adapter must not raise on normal MA-BBOB inputs (the `np.Inf`
  fix removes the one guaranteed crash).
- The coverage script captures coverage up to any mid-run failure and still
  reports it.

## Explicitly out of scope
- No change to `ch6_discussion.tex:79` (future-work loop-gating idea).
- No re-running of the LLaMEA-SAGE generation loop; no LLM in the loop.
- No modification of existing experiment result files or existing experiment
  config semantics.
