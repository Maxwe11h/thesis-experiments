# Repository Reorganization — Design

**Date:** 2026-06-07
**Status:** Approved (design); pending implementation plan
**Goal:** Reorganize the thesis repository into clear top-level categories — experiment
code, data-analysis code, results data, thesis-figure scripts, and the thesis itself —
without losing data, and make the experiments clearly reusable and rerunnable for
anyone new to the codebase.

## Motivation

The repository works but is hard to navigate for a newcomer:

- ~14 `run_*.py` / `run_*.sh` / `setup_*.sh` drivers are scattered at the repo root.
- Stray LaTeX build artifacts (`thesis.aux/.fls/.log/.out/.fdb_latexmk`) sit at root;
  the real thesis lives in `docs/thesisLatex/`.
- `analysis/` mixes three concerns: analysis notebooks, an analysis package
  (`analysis/phase4/`), and figure-export scripts (`export_*.py`).
- `docs/` is a catch-all (markdown docs, reference PDFs, winner algorithms, the thesis,
  zips, specs).
- There is **no README, `requirements.txt`, or `environment.yml`** anywhere — the main
  reproducibility gap.

## Decisions (from brainstorming)

- **Scope:** full restructure into category-mirrored top-level directories (Approach A).
- **Results data:** the 7 `results*` dirs are large and gitignored; the 4 that feed the
  thesis stay in place. The 3 unused dirs move to a gitignored `results_archive/`.
- **Thesis:** promoted to a top-level `thesis/` (from `docs/thesisLatex/`).
- **Reproducibility:** add a top-level `README.md` + runbook, and a best-effort
  `requirements.txt` derived from imports.
- **History:** use `git mv` for tracked files to preserve history.

### Results-directory audit

Traced every reference across `analysis/`, `experiments/`, and the figure-export pipeline:

| dir | size | feeds thesis? | used by |
|---|---|---|---|
| `results_phase1` | 119M | yes | `export_figures.py`, phase-1 notebooks |
| `results_phase3` | 407M | yes | `export_figures.py`, `phase3_feedback_analysis.ipynb` |
| `results_phase4` | 686M | yes | `export_phase4_figures.py`, `phase4_analysis.ipynb` |
| `results_phase4_full_suite` | 483M | yes | `export_phase4_full_suite_figures.py` |
| `results/` | 17M | no | only the old `run_experiment.py` / `run_vanilla_baseline.py` output (discarded feature-selection experiment) |
| `results_phase4_partial` | 31M | no | referenced nowhere (superseded partial Phase 4 run) |
| `results_token_test` | 6.5M | no | only `experiments/phase4_token_test.py` (one-off token-budget test) |

**4 feed the thesis (stay in place); 3 are unused (→ `results_archive/`).**

### Phase 4 vs. Phase 4 full suite (for the README)

- **Phase 4 (`results_phase4`)** — *design & selection*. LLM-driven LLaMEA evolution of
  algorithms under 4 feedback conditions (`vanilla`, `neutral`, `sage`,
  `combined_neutral`) on a curated 20-instance MA-BBOB *training* set, 500 candidates,
  10 seeds. Produces the winner `.py` files in `docs/stage4_winners/`.
- **Phase 4 full suite (`results_phase4_full_suite`)** — *generalization test*. No LLM:
  freezes the 4 winners + 2 external baselines (CMA-ES, `neighborhood_adaptive_de`) and
  benchmarks them on all 1000 MA-BBOB functions × 5 instances × {5,10,20} dims at a
  2000×dim budget.

## Target top-level structure

```
thesis/  (repo root)
├── README.md                  ← NEW: overview, repo map, pipeline table, quickstart
├── requirements.txt           ← NEW: best-effort, derived from imports
├── .gitignore                 ← updated
│
├── experiments/               ── EXPERIMENT CODE
│   ├── config.py, mabbob_problem.py, feedback.py, trajectory_logger.py,
│   │   initial_population.py, model_selection.py, run_experiment.py,
│   │   phase{1,3,4}_config.py, phase4_full_suite_config.py,
│   │   phase{1,3,4}_experiment.py, phase4_full_suite_runner.py,
│   │   phase4_token_test.py, benchmark_eval_overhead.py
│   └── run/                    ← NEW package: entry-point drivers (from root)
│       ├── __init__.py
│       ├── run_phase1.py, run_phase1_gemini.sh
│       ├── run_phase3.py, run_phase3.sh, run_phase3_top5.sh
│       ├── run_phase4.py, run_phase4_full_suite.py
│       └── legacy/             ← discarded feature-selection drivers
│           run_conditions.py, run_baseline_comparison.py,
│           run_model_selection.py, run_vanilla_baseline.py
│
├── analysis/                  ── DATA ANALYSIS CODE
│   ├── *.ipynb (phase1/3/4, qualitative, robustness, …)
│   ├── phase4/ (code_identity, failure_modes, steering)
│   ├── winner_coverage.py, benchmark_features.py, run_phase1_classifier.py
│   └── figs_phase1/, figs_phase4/, figs_phase4_full_suite/  (tracked CSV/PDF intermediates)
│
├── figures/                   ── THESIS-FIGURE SCRIPTS (NEW top-level)
│   ├── export_figures.py
│   ├── export_phase4_figures.py
│   └── export_phase4_full_suite_figures.py
│
├── thesis/                    ── THE THESIS (NEW, from docs/thesisLatex)
│   ├── thesis.tex, titlepage.tex, bibliography.bib, proposal.tex
│   ├── chapters/, appendices/, figures/
│   └── revision_plan_*.md, supervisor_notes_*.md
│
├── baselines/                 (unchanged — referenced by full_suite_config)
├── slurm/                     (+ setup_server.sh, setup_vibranium_gemini.sh; invocations updated)
├── tests/                     (+ test_gemini.py)
├── docs/                      (markdown docs, sources/ PDFs, stage4_winners/, superpowers/, zips)
│
├── results_phase1/  results_phase3/  results_phase4/  results_phase4_full_suite/   ← in place (gitignored)
├── results_archive/           ← NEW (gitignored): results/, results_phase4_partial/, results_token_test/
│
├── BLADE/   LLaMEA/           (submodules, untouched)
```

## Moves and the path edits each triggers

1. **Run drivers** `run_*.py|sh` (root) → `experiments/run/` (legacy ones → `run/legacy/`).
   - `experiments/run/` becomes a package (`__init__.py`) and drivers are invoked via
     `python -m experiments.run.run_phaseX` from the repo root. This preserves the
     existing absolute imports (`from experiments.<mod> import …`), which would break
     under direct `python experiments/run/run_phaseX.py` script invocation (sys.path[0]
     would be the script dir, not the repo root).
   - **Edits:** all `python run_*.py` calls in `slurm/*.sbatch`, `slurm/*.sh`, and the
     `.sh` wrappers → `python -m experiments.run.…`.
   - **Verify during execution:** that the 4 legacy drivers do not import each other by
     bare module name (e.g. `from run_experiment import`); if they do, fix to absolute
     `from experiments.run_experiment import`.

2. **Figure scripts** `analysis/export_*.py` (3 files) → `figures/`.
   - `REPO_ROOT = SCRIPT_DIR.parent` stays valid (still one level below root); inputs
     (`results_*`, `analysis/figs_*`) resolve unchanged.
   - **Edits:** the output path constant in all 3 (`FIGURES_DIR` / `THESIS_FIG_DIR`):
     `docs/thesisLatex/figures` → `thesis/figures`.

3. **Thesis** `docs/thesisLatex/` → `thesis/` (moved as a unit; internal `\input` and
   `figures/` relative paths stay intact). Proposal + revision/supervisor notes move with it.

4. **Setup scripts** `setup_server.sh`, `setup_vibranium_gemini.sh` → `slurm/`.
   `test_gemini.py` → `tests/`.

5. **Unused results** → `results_archive/`: `results/`, `results_phase4_partial/`,
   `results_token_test/` (plain `mv`, gitignored). The drivers that produce these
   (the legacy feature/model-selection runners and the one-off
   `experiments/phase4_token_test.py`) are **left unedited** — respecting
   `feedback_dont_modify_existing` and keeping all archived-data drivers uniform. They
   are discarded/one-off, so rerunning them recreates a gitignored dir at the repo root;
   this is documented in the README.

**Staying put (avoid churn):** `docs/stage4_winners/` (referenced by
`phase4_full_suite_config.py`, `winner_coverage.py`, and `ch5_results.tex`),
`baselines/`, the 4 active `results_phase*` dirs, `analysis/figs_*`.

## Cleanup

- Delete stray root LaTeX leftovers: `thesis.aux`, `thesis.fdb_latexmk`, `thesis.fls`,
  `thesis.log`, `thesis.out` (untracked).
- `.gitignore` updates:
  - `docs/thesisLatex/thesis.pdf` → `thesis/thesis.pdf`
  - add `results_archive/`
  - add `.coverage`

## Reproducibility deliverables

- **`README.md`** (root):
  - one-paragraph project summary;
  - the repo map (tree above);
  - a pipeline table: *Phase → driver (`experiments/run/…`) → SLURM script → results dir
    → analysis notebook → export script → thesis figures*;
  - quickstart: clone with `--recurse-submodules`, create the conda env, set `.env`
    (API keys, `OLLAMA_PORT`), run a phase via `python -m experiments.run.…`, regenerate
    figures via `figures/export_*.py`, build the thesis via `latexmk` in `thesis/`;
  - pointers to `docs/server_runbook.md` and `docs/slurm_guide.md`.
- **`requirements.txt`** best-effort, derived from actual imports across
  `experiments/`, `analysis/`, `figures/` (clearly marked best-effort; the BLADE and
  LLaMEA submodules carry their own dependency specs).

## Mechanics & verification

- Use `git mv` for all tracked files (preserves history); plain `mv` for gitignored data.
- After all moves and edits:
  1. import smoke-test (`python -c "import experiments.phase4_full_suite_config"` etc.);
  2. run the **3 figure-export scripts** — confirms every rewritten input/output path
     against the on-disk results dirs;
  3. `pytest tests/`;
  4. `latexmk` build of `thesis/` to confirm `\input` and figure paths resolve;
  5. `git status` / `git log --follow` spot-check to confirm history is preserved.

## Out of scope

- No changes to experiment logic, configs, or results data content (per
  `feedback_dont_modify_existing`: reorganization moves files but does not edit experiment
  configs' behavior).
- Submodules `BLADE/` and `LLaMEA/` are untouched.
- No deletion of any results data (the 3 unused dirs are archived, not deleted).
```
