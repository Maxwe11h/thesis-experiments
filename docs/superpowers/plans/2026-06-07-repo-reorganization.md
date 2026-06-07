# Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the thesis repo into clear top-level categories (experiment code, analysis code, results data, thesis-figure scripts, the thesis) without losing data, and add a README + requirements so it is reusable/rerunnable.

**Architecture:** Pure file/structure reorganization via `git mv` (history-preserving) plus mechanical path-reference edits in code, SLURM scripts, and docs. No experiment logic changes. The run drivers become a package (`experiments/run/`) invoked as `python -m experiments.run.<name>` from the repo root.

**Tech Stack:** Python 3.11, git submodules (BLADE, LLaMEA), SLURM shell scripts, LaTeX (latexmk).

**Spec:** `docs/superpowers/specs/2026-06-07-repo-reorganization-design.md`

> **Deviation from spec (intentional):** The spec said to repoint `experiments/phase4_token_test.py`'s output dir. We will **not** edit it (respects `feedback_dont_modify_existing` and keeps all archived-data drivers uniform). Its data is archived; rerunning the one-off recreates `results_token_test/` at root, which is gitignored and documented in the README.

> **Environment caveat:** Runtime smoke checks (import, `--list`, export scripts, `pytest`, `latexmk`) require the conda env with BLADE+LLaMEA installed. If executing on a machine without that env, perform the **structural** checks (git status/log, grep for stale references) and defer runtime smokes, noting which were deferred.

> **Execution deviations (what actually happened):** Verification surfaced fixes beyond the planned tasks, each in its own commit:
> - `test_gemini.py` was renamed to **`tests/check_gemini.py`** (not `tests/test_gemini.py` as written below): the script has unguarded top-level `sys.exit(1)`, so pytest auto-discovery (`test_*.py`) aborted collection. Renaming keeps it in `tests/` per the spec's intent while excluding it from the suite.
> - `figures/export_phase4_full_suite_figures.py` `LOCAL_FIG_DIR` was repointed from `SCRIPT_DIR/...` to `REPO_ROOT/"analysis"/"figs_phase4_full_suite"` (the script-relative path moved with the script); an accidental `figures/figs_phase4_full_suite/` commit was removed.
> - `figures/export_phase4_figures.py` cross-module import fixed: `from analysis.export_figures` → `from figures.export_figures`.
> - The latexmk build re-created `thesis/thesis.pdf`, which was committed before `.gitignore` was repointed; it was untracked (`git rm --cached`).
> - Usage examples in `experiments/**.py` docstrings and the two `slurm/setup_*.sh` scripts were also updated to `python -m experiments.run.*` (the planned Task 9 covered only `docs/*.md`).
> - Final review: APPROVED_WITH_NOTES; pytest 35 passed / 1 failed (the 1 failure is `TestIntegrationOllama` needing a live Ollama model — pre-existing env dependency, unrelated to the reorg).

---

## File structure after reorganization

```
experiments/run/__init__.py                         (new, empty)
experiments/run/run_phase1.py                        (from ./run_phase1.py)
experiments/run/run_phase1_gemini.sh                 (from ./run_phase1_gemini.sh)
experiments/run/run_phase3.py                        (from ./run_phase3.py)
experiments/run/run_phase3.sh                        (from ./run_phase3.sh)
experiments/run/run_phase3_top5.sh                   (from ./run_phase3_top5.sh)
experiments/run/run_phase4.py                        (from ./run_phase4.py)
experiments/run/run_phase4_full_suite.py             (from ./run_phase4_full_suite.py)
experiments/run/legacy/__init__.py                   (new, empty)
experiments/run/legacy/run_conditions.py             (from ./run_conditions.py)
experiments/run/legacy/run_baseline_comparison.py    (from ./run_baseline_comparison.py)
experiments/run/legacy/run_model_selection.py        (from ./run_model_selection.py)
experiments/run/legacy/run_vanilla_baseline.py       (from ./run_vanilla_baseline.py)
figures/export_figures.py                            (from analysis/export_figures.py)
figures/export_phase4_figures.py                     (from analysis/export_phase4_figures.py)
figures/export_phase4_full_suite_figures.py          (from analysis/export_phase4_full_suite_figures.py)
thesis/                                              (from docs/thesisLatex/)
slurm/setup_server.sh                                (from ./setup_server.sh)
slurm/setup_vibranium_gemini.sh                      (from ./setup_vibranium_gemini.sh)
tests/test_gemini.py                                 (from ./test_gemini.py)
results_archive/results/                             (from ./results/, gitignored)
results_archive/results_phase4_partial/              (from ./results_phase4_partial/, gitignored)
results_archive/results_token_test/                  (from ./results_token_test/, gitignored)
README.md                                            (new)
requirements.txt                                     (new)
.env.example                                         (new)
.gitignore                                           (modified)
```

Edited in place (path-reference fixes only): the 3 `figures/export_*.py` output constants; `experiments/run/legacy/run_baseline_comparison.py` + `run_vanilla_baseline.py` root-path hacks; the 3 `.sh` wrappers' internal python calls; ~15 SLURM invocations; ~30 doc command examples in 6 markdown files.

---

## Task 0: Safety checkpoint, hygiene, and working branch

**Files:** Modify `.gitignore`; delete untracked root LaTeX artifacts; commit the design docs.

> Why up front: later tasks use `git add -A`. We must first (a) ignore `.coverage` and
> (b) commit the untracked spec/plan, so those don't get swept into unrelated commits.

- [ ] **Step 1: Confirm a clean tree**

Run: `git status --porcelain`
Expected: only untracked items — `?? .coverage` and the `?? docs/superpowers/specs/2026-06-07-...md` / `?? docs/superpowers/plans/2026-06-07-...md` design docs. No staged/modified tracked files.

- [ ] **Step 2: Create the working branch**

```bash
git checkout -b chore/repo-reorg
```

- [ ] **Step 3: Delete stray repo-root LaTeX build artifacts** (untracked, regenerable; distinct from the ones inside `docs/thesisLatex/`)

```bash
rm -f thesis.aux thesis.fdb_latexmk thesis.fls thesis.log thesis.out
```
Verify: `ls thesis.aux thesis.fls thesis.log thesis.out thesis.fdb_latexmk 2>/dev/null; echo done`
Expected: prints only `done`.

- [ ] **Step 4: Ignore `.coverage` early**

```bash
grep -qxF '.coverage' .gitignore || printf '\n# Coverage data\n.coverage\n' >> .gitignore
git check-ignore .coverage
```
Expected: `git check-ignore` prints `.coverage` (now ignored).

- [ ] **Step 5: Commit the `.gitignore` change and the design docs together**

```bash
git add .gitignore docs/superpowers/specs/2026-06-07-repo-reorganization-design.md docs/superpowers/plans/2026-06-07-repo-reorganization.md
git commit -m "docs: add repo-reorg spec and plan; ignore .coverage"
```

- [ ] **Step 6: Snapshot the current layout for reference**

```bash
git ls-files | sort > /tmp/reorg_tracked_before.txt
wc -l /tmp/reorg_tracked_before.txt
```
Expected: prints the count of tracked files (record it; the same files must exist after, at new paths). The spec/plan are now tracked, so they're already in this baseline.

---

## Task 1: Create directory skeleton

**Files:**
- Create: `experiments/run/__init__.py`, `experiments/run/legacy/__init__.py`

- [ ] **Step 1: Make new directories and package markers**

```bash
mkdir -p experiments/run/legacy figures results_archive
touch experiments/run/__init__.py experiments/run/legacy/__init__.py
```

- [ ] **Step 2: Verify**

Run: `ls experiments/run experiments/run/legacy && ls -d figures results_archive`
Expected: `__init__.py` present in both package dirs; `figures` and `results_archive` exist.

- [ ] **Step 3: Commit**

```bash
git add experiments/run/__init__.py experiments/run/legacy/__init__.py
git commit -m "chore: scaffold experiments/run package and figures dir"
```

---

## Task 2: Move the thesis to top-level `thesis/`

**Files:**
- Move: `docs/thesisLatex/` → `thesis/`
- Delete (gitignored build artifacts, regenerable)

- [ ] **Step 1: Remove regenerable LaTeX build artifacts in the source dir**

```bash
rm -f docs/thesisLatex/thesis.aux docs/thesisLatex/thesis.bbl docs/thesisLatex/thesis.blg \
      docs/thesisLatex/thesis.fdb_latexmk docs/thesisLatex/thesis.fls docs/thesisLatex/thesis.log \
      docs/thesisLatex/thesis.out docs/thesisLatex/thesis.synctex.gz docs/thesisLatex/thesis.pdf
```

- [ ] **Step 2: Move the tracked thesis tree (preserves history)**

```bash
git mv docs/thesisLatex thesis
```

- [ ] **Step 3: Verify nothing was left behind**

Run: `test -d docs/thesisLatex && echo "LEFTOVER" || echo "clean"; ls thesis/thesis.tex thesis/chapters thesis/figures`
Expected: prints `clean`, and the thesis files exist under `thesis/`.

- [ ] **Step 4: (Runtime, if env available) Confirm the thesis still builds**

```bash
cd thesis && latexmk -pdf -interaction=nonstopmode thesis.tex >/tmp/latexmk.log 2>&1; echo "exit=$?"; cd ..
```
Expected: `exit=0` and `thesis/thesis.pdf` produced. If LaTeX is unavailable, skip and note as deferred.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: promote thesis from docs/thesisLatex to top-level thesis/"
```

---

## Task 3: Move figure-export scripts to `figures/` and fix output paths

**Files:**
- Move: `analysis/export_figures.py`, `analysis/export_phase4_figures.py`, `analysis/export_phase4_full_suite_figures.py` → `figures/`
- Modify: output-path constants in all three (`docs/thesisLatex/figures` → `thesis/figures`)

- [ ] **Step 1: Move the three scripts**

```bash
git mv analysis/export_figures.py figures/export_figures.py
git mv analysis/export_phase4_figures.py figures/export_phase4_figures.py
git mv analysis/export_phase4_full_suite_figures.py figures/export_phase4_full_suite_figures.py
```

- [ ] **Step 2: Rewrite the output-path constant in each**

```bash
sed -i '' 's#"docs" / "thesisLatex" / "figures"#"thesis" / "figures"#g' \
  figures/export_figures.py figures/export_phase4_figures.py figures/export_phase4_full_suite_figures.py
```

- [ ] **Step 3: Verify the rewrite and that no `docs/thesisLatex` reference remains**

Run: `grep -nE "thesis. / .figures|thesisLatex" figures/*.py`
Expected: each file shows `REPO_ROOT / "thesis" / "figures"` (constant `FIGURES_DIR` / `THESIS_FIG_DIR`); **no** `thesisLatex` matches.

- [ ] **Step 4: (Runtime, if env available) Smoke-run the smallest exporter**

Run: `python figures/export_phase4_full_suite_figures.py >/tmp/exp_fs.log 2>&1; echo "exit=$?"; ls thesis/figures/fig_phase4_fullsuite_*.pdf`
Expected: `exit=0`; `fig_phase4_fullsuite_*.pdf` written under `thesis/figures/`. (Reads `results_phase4_full_suite/`, which stays in place.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: move figure-export scripts to figures/, point output at thesis/figures"
```

---

## Task 4: Move run drivers into `experiments/run/` (+ `legacy/`) and fix imports

**Files:**
- Move active: `run_phase1.py`, `run_phase1_gemini.sh`, `run_phase3.py`, `run_phase3.sh`, `run_phase3_top5.sh`, `run_phase4.py`, `run_phase4_full_suite.py` → `experiments/run/`
- Move legacy: `run_conditions.py`, `run_baseline_comparison.py`, `run_model_selection.py`, `run_vanilla_baseline.py` → `experiments/run/legacy/`
- Modify: root-path hacks in `run_baseline_comparison.py` + `run_vanilla_baseline.py`; internal python calls in the 3 `.sh` wrappers

- [ ] **Step 1: Move active drivers**

```bash
git mv run_phase1.py experiments/run/run_phase1.py
git mv run_phase1_gemini.sh experiments/run/run_phase1_gemini.sh
git mv run_phase3.py experiments/run/run_phase3.py
git mv run_phase3.sh experiments/run/run_phase3.sh
git mv run_phase3_top5.sh experiments/run/run_phase3_top5.sh
git mv run_phase4.py experiments/run/run_phase4.py
git mv run_phase4_full_suite.py experiments/run/run_phase4_full_suite.py
```

- [ ] **Step 2: Move legacy drivers**

```bash
git mv run_conditions.py experiments/run/legacy/run_conditions.py
git mv run_baseline_comparison.py experiments/run/legacy/run_baseline_comparison.py
git mv run_model_selection.py experiments/run/legacy/run_model_selection.py
git mv run_vanilla_baseline.py experiments/run/legacy/run_vanilla_baseline.py
```

- [ ] **Step 3: Fix the repo-root hack in the two legacy drivers that compute `ROOT` from `__file__`**

These now live three levels below the repo root (`experiments/run/legacy/`). Replace the `ROOT = ...` line in each so it resolves to the repo root.

```bash
sed -i '' 's#^ROOT = os.path.dirname(os.path.abspath(__file__))#ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))#' \
  experiments/run/legacy/run_baseline_comparison.py experiments/run/legacy/run_vanilla_baseline.py
```

Verify:
Run: `grep -n "^ROOT = " experiments/run/legacy/run_baseline_comparison.py experiments/run/legacy/run_vanilla_baseline.py`
Expected: each shows `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))`.

- [ ] **Step 4: Update internal python calls in the moved `.sh` wrappers**

```bash
sed -i '' 's#python run_phase1\.py#python -m experiments.run.run_phase1#g' experiments/run/run_phase1_gemini.sh
sed -i '' 's#python run_phase3\.py#python -m experiments.run.run_phase3#g' experiments/run/run_phase3.sh experiments/run/run_phase3_top5.sh
```

Verify:
Run: `grep -nE "run_phase[0-9]\.py" experiments/run/*.sh`
Expected: no matches.

- [ ] **Step 5: (Runtime, if env available) Import + functional smoke for every driver**

```bash
for m in run_phase1 run_phase3 run_phase4 run_phase4_full_suite \
         legacy.run_conditions legacy.run_baseline_comparison legacy.run_model_selection legacy.run_vanilla_baseline; do
  python -c "import importlib; importlib.import_module('experiments.run.$m')" && echo "import ok: $m" || echo "IMPORT FAIL: $m"
done
python -m experiments.run.run_phase1 --list >/dev/null 2>&1 && echo "phase1 --list ok"
python -m experiments.run.run_phase4 --list >/dev/null 2>&1 && echo "phase4 --list ok"
python -m experiments.run.run_phase4_full_suite --help >/dev/null 2>&1 && echo "full_suite --help ok"
```
Expected: `import ok` for all 8 modules; the three CLI smokes print `ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: move run drivers into experiments/run (legacy/ for discarded ones), fix paths"
```

---

## Task 5: Update SLURM invocations

**Files (modify):** `slurm/phase1_ollama.sbatch`, `slurm/phase1_vllm.sbatch`, `slurm/phase3.sbatch`, `slurm/phase3_all.sbatch`, `slurm/phase3_remaining.sbatch`, `slurm/phase4.sbatch`, `slurm/phase4_vanilla.sbatch`, `slurm/phase4_neutral.sbatch`, `slurm/phase4_sage.sbatch`, `slurm/phase4_directional.sbatch`, `slurm/phase4_combined_neutral.sbatch`, `slurm/phase4_full_suite.sh`, `slurm/phase4_full_suite_optA.sh`, `slurm/phase4_full_suite_nade.sh`, `slurm/phase4_full_suite_smoke.sh`

- [ ] **Step 1: Rewrite all four driver patterns across `slurm/`**

```bash
sed -i '' 's#python run_phase1\.py#python -m experiments.run.run_phase1#g' slurm/*.sbatch slurm/*.sh
sed -i '' 's#python run_phase3\.py#python -m experiments.run.run_phase3#g' slurm/*.sbatch slurm/*.sh
sed -i '' 's#python run_phase4\.py#python -m experiments.run.run_phase4#g' slurm/*.sbatch slurm/*.sh
sed -i '' 's#python run_phase4_full_suite\.py#python -m experiments.run.run_phase4_full_suite#g' slurm/*.sbatch slurm/*.sh
```

(The last pattern also correctly rewrites `$ENV/bin/python run_phase4_full_suite.py` → `$ENV/bin/python -m experiments.run.run_phase4_full_suite`.)

- [ ] **Step 2: Verify no stale invocation remains**

Run: `grep -rnE "run_phase[0-9_a-z]*\.py" slurm/`
Expected: no matches.

- [ ] **Step 3: Verify the new invocations are present and well-formed**

Run: `grep -rnE "experiments\.run\.run_phase" slurm/ | wc -l`
Expected: 17 (matches the number of original `python run_phase*.py` invocations in `slurm/`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: update SLURM scripts to python -m experiments.run.* invocations"
```

---

## Task 6: Move setup scripts and the Gemini connectivity test

**Files:**
- Move: `setup_server.sh`, `setup_vibranium_gemini.sh` → `slurm/`; `test_gemini.py` → `tests/`

- [ ] **Step 1: Move the files**

```bash
git mv setup_server.sh slurm/setup_server.sh
git mv setup_vibranium_gemini.sh slurm/setup_vibranium_gemini.sh
git mv test_gemini.py tests/test_gemini.py
```

- [ ] **Step 2: Verify root no longer has stray scripts**

Run: `ls run_*.py run_*.sh setup_*.sh test_gemini.py 2>/dev/null; echo "done"`
Expected: prints only `done` (no matches) — root is clear of these scripts.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: move setup scripts to slurm/ and test_gemini.py to tests/"
```

---

## Task 7: Archive the three unused results directories

**Files:** (all gitignored — plain `mv`, not `git mv`)
- Move: `results/`, `results_phase4_partial/`, `results_token_test/` → `results_archive/`

- [ ] **Step 1: Confirm they are untracked before moving (no data is in git)**

Run: `git ls-files results results_phase4_partial results_token_test | wc -l`
Expected: `0` (nothing tracked — safe to move on disk).

- [ ] **Step 2: Move them**

```bash
[ -d results ] && mv results results_archive/results
[ -d results_phase4_partial ] && mv results_phase4_partial results_archive/results_phase4_partial
[ -d results_token_test ] && mv results_token_test results_archive/results_token_test
```

- [ ] **Step 3: Verify**

Run: `ls results_archive && ls -d results results_phase4_partial results_token_test 2>/dev/null; echo "done"`
Expected: the three dirs appear under `results_archive/`; the root copies are gone (only `done` after the second `ls`).

- [ ] **Step 4: Confirm the four active results dirs are untouched**

Run: `ls -d results_phase1 results_phase3 results_phase4 results_phase4_full_suite`
Expected: all four still present at root.

(No commit — these are gitignored. The `.gitignore` change is Task 8.)

---

## Task 8: Update `.gitignore`

**Files:** Modify `.gitignore`

- [ ] **Step 1: Repoint the thesis PDF ignore and add the new ignores**

```bash
sed -i '' 's#^docs/thesisLatex/thesis.pdf#thesis/thesis.pdf#' .gitignore
```

- [ ] **Step 2: Append `results_archive/` (only if not already present)**

```bash
grep -qxF 'results_archive/' .gitignore || printf '\n# Archived (superseded/one-off) results\nresults_archive/\n' >> .gitignore
```
(`.coverage` was already added in Task 0.)

- [ ] **Step 3: Verify ignores are effective**

Run: `git check-ignore -v thesis/thesis.pdf results_archive/results .coverage`
Expected: each path prints a matching `.gitignore` rule.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for thesis/, results_archive/, .coverage"
```

---

## Task 9: Update documentation command examples

**Files (modify):** `docs/phase1_README.md`, `docs/phase3_conditions.md`, `docs/slurm_guide.md`, `docs/model_selection_experiment.md`, `docs/server_runbook.md`, `docs/feature_selection_experiment.md`

- [ ] **Step 1: Active-driver docs (phase1/phase3/slurm)**

```bash
sed -i '' 's#python run_phase1\.py#python -m experiments.run.run_phase1#g' docs/phase1_README.md docs/slurm_guide.md
sed -i '' 's#python run_phase3\.py#python -m experiments.run.run_phase3#g' docs/phase3_conditions.md
```

- [ ] **Step 2: Legacy-driver docs (model selection, feature selection, server runbook)**

```bash
sed -i '' 's#python run_model_selection\.py#python -m experiments.run.legacy.run_model_selection#g' docs/model_selection_experiment.md
sed -i '' 's#python run_conditions\.py#python -m experiments.run.legacy.run_conditions#g' docs/server_runbook.md docs/feature_selection_experiment.md
sed -i '' 's#^  run_conditions\.py#  experiments/run/legacy/run_conditions.py#' docs/feature_selection_experiment.md
```

- [ ] **Step 3: Verify no stale `run_*.py` invocation remains in docs**

Run: `grep -rnE "python run_(phase[0-9]|conditions|model_selection|baseline_comparison|vanilla)\.py" docs/`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: update command examples to python -m experiments.run.* paths"
```

---

## Task 10: Add `.env.example`

**Files:** Create `.env.example`

- [ ] **Step 1: Write the example env file** (real `.env` stays gitignored)

Create `.env.example`:

```bash
# Copy to .env (gitignored) and fill in your values.

# Google GenAI / Gemini API key — used by Phase 1 Gemini models and
# the Phase 3/4 gemini-3-flash conditions.
GEMINI_API_KEY=

# Ollama server port. Default 11434; use 11435 for a second GPU instance
# (see docs/server_runbook.md).
OLLAMA_PORT=11434
```

- [ ] **Step 2: Verify it is tracked but real `.env` is not**

Run: `git check-ignore .env; git check-ignore .env.example || echo ".env.example NOT ignored (good)"`
Expected: `.env` prints (ignored); `.env.example` prints the "NOT ignored (good)" message.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add .env.example documenting required environment variables"
```

---

## Task 11: Add `requirements.txt`

**Files:** Create `requirements.txt`

- [ ] **Step 1: Write the best-effort requirements file**

Create `requirements.txt`:

```
# Best-effort top-level Python dependencies, derived from imports across
# experiments/, analysis/, and figures/. Versions intentionally unpinned;
# Python >=3.11 is required (matches BLADE/LLaMEA).
#
# The BLADE and LLaMEA submodules carry their own dependency specs — install
# them in editable mode after these:
#     pip install -e ./BLADE -e ./LLaMEA
#
# Core scientific stack
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn

# Benchmarking / optimization
ioh
cma
pyarrow

# LLM API
google-genai
```

- [ ] **Step 2: (Runtime, if env available) Sanity-check the listed packages import**

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, ioh, cma, pyarrow; from google import genai; print('deps ok')"
```
Expected: `deps ok`. If a package name differs in the active env, correct the file to match.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "docs: add best-effort requirements.txt"
```

---

## Task 12: Add top-level `README.md`

**Files:** Create `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:

````markdown
# Behavioural Feedback for LLM-Driven Metaheuristic Design

Bachelor's thesis (LIACS, Leiden) studying whether **behavioural feedback** improves
LLM-driven algorithm design with [LLaMEA](https://github.com/Maxwe11h/LLaMEA), benchmarked
via [BLADE](https://github.com/Maxwe11h/BLADE) on MA-BBOB. This repo holds the experiment
code, analysis, figure scripts, and the thesis itself.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Experiment code: configs, MA-BBOB problem, feedback functions, phase experiments. |
| `experiments/run/` | Entry-point drivers, invoked as `python -m experiments.run.<name>`. `legacy/` holds discarded feature/model-selection drivers. |
| `analysis/` | Analysis notebooks + the `analysis/phase4/` package; tracked intermediates in `analysis/figs_*`. |
| `figures/` | Scripts that render the thesis figures into `thesis/figures/`. |
| `thesis/` | The LaTeX thesis (`thesis.tex`, `chapters/`, `figures/`, `bibliography.bib`). |
| `baselines/` | External baseline algorithms (e.g. NADE, the 2025 MA-BBOB winner). |
| `slurm/` | SLURM batch + node/server setup scripts. |
| `tests/` | Test suite. |
| `docs/` | Markdown docs, reference PDFs (`sources/`), winner algorithms (`stage4_winners/`), design specs/plans. |
| `BLADE/`, `LLaMEA/` | Git submodules. |
| `results_phase{1,3,4}/`, `results_phase4_full_suite/` | Experiment outputs (large, **gitignored**). |
| `results_archive/` | Superseded/one-off results (`results/`, `results_phase4_partial/`, `results_token_test/`), gitignored. |

## Experiment pipeline

| Stage | Driver | SLURM | Results dir | Analysis | Figures |
|---|---|---|---|---|---|
| Phase 1 — model screening | `python -m experiments.run.run_phase1` | `slurm/phase1_{ollama,vllm}.sbatch` | `results_phase1/` | `analysis/phase1_*.ipynb` | `figures/export_figures.py` |
| Phase 3 — feedback screening | `python -m experiments.run.run_phase3` | `slurm/phase3*.sbatch` | `results_phase3/` | `analysis/phase3_feedback_analysis.ipynb` | `figures/export_figures.py` |
| Phase 4 — design & selection | `python -m experiments.run.run_phase4` | `slurm/phase4*.sbatch` | `results_phase4/` | `analysis/phase4_analysis.ipynb`, `analysis/phase4/` | `figures/export_phase4_figures.py` |
| Phase 4 — full-suite generalization | `python -m experiments.run.run_phase4_full_suite` | `slurm/phase4_full_suite*.sh` | `results_phase4_full_suite/` | `analysis/winner_coverage.py` | `figures/export_phase4_full_suite_figures.py` |

**Phase 4 vs. full suite:** Phase 4 runs the LLM (LLaMEA) to *design and select* the best
algorithm per feedback condition on a curated 20-instance training set, producing the
winners in `docs/stage4_winners/`. The full suite then *freezes those winners* (plus
external baselines) and tests *generalization* on all 1000 MA-BBOB functions — no LLM.

## Quickstart

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url> thesis
cd thesis

# 2. Environment (Python >= 3.11)
conda create -n thesis python=3.11 -y && conda activate thesis
pip install -r requirements.txt
pip install -e ./BLADE -e ./LLaMEA

# 3. Configure secrets (real .env is gitignored)
cp .env.example .env   # then set GEMINI_API_KEY (and OLLAMA_PORT if using Ollama)

# 4. Run an experiment (always from the repo root)
python -m experiments.run.run_phase1 --list

# 5. Regenerate thesis figures
python figures/export_figures.py
python figures/export_phase4_figures.py
python figures/export_phase4_full_suite_figures.py

# 6. Build the thesis
cd thesis && latexmk -pdf thesis.tex
```

Run all drivers **from the repo root** so the `experiments` package imports and the
relative `results_*` paths resolve. For HPC, see `docs/slurm_guide.md`; for server
operations (Ollama ports, GPU layout), see `docs/server_runbook.md`.

## Notes

- `.env` is never committed. `results_*` directories are gitignored (large generated data).
- `results_archive/` holds superseded data; its drivers (the legacy ones and the one-off
  `experiments/phase4_token_test.py`) still default to writing at the repo root if rerun.
````

- [ ] **Step 2: Verify the README references resolve**

Run: `for p in experiments/run figures thesis baselines slurm tests docs results_archive; do test -e $p && echo "ok $p" || echo "MISSING $p"; done`
Expected: `ok` for every path.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add top-level README with layout, pipeline, and quickstart"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Root is decluttered**

Run: `ls *.py *.sh 2>/dev/null; echo "---"; ls thesis.aux thesis.fls thesis.log thesis.out thesis.fdb_latexmk 2>/dev/null; echo done`
Expected: no stray `run_*`/`setup_*`/`test_gemini.py` at root; no stray `thesis.*` build artifacts; prints `done`.

- [ ] **Step 2: No tracked file was lost (only moved)**

```bash
git ls-files | sort > /tmp/reorg_tracked_after.txt
echo "before: $(wc -l < /tmp/reorg_tracked_before.txt)  after: $(wc -l < /tmp/reorg_tracked_after.txt) (after = before + new files: README, requirements.txt, .env.example, 2 __init__.py)"
```
Expected: `after` = `before` + 5 (the only genuinely new tracked files).

- [ ] **Step 3: History preserved across a representative move**

Run: `git log --follow --oneline -- experiments/run/run_phase1.py | head -3`
Expected: shows commits predating this reorg (history followed through the rename).

- [ ] **Step 4: No stale references anywhere**

Run:
```bash
grep -rnE "docs/thesisLatex|run_phase[0-9_a-z]*\.py|analysis/export_" slurm/ docs/ figures/ experiments/run/ 2>/dev/null \
  | grep -v "experiments.run." | grep -v "Binary"
```
Expected: no matches (every reference now uses the new paths/invocations).

- [ ] **Step 5: (Runtime, if env available) Test suite passes**

Run: `pytest tests/ -q 2>/tmp/pytest.log; echo "exit=$?"`
Expected: `exit=0`. If the env is unavailable, note as deferred.

- [ ] **Step 6: (Runtime, if env available) All three exporters succeed end-to-end**

```bash
python figures/export_figures.py >/tmp/e1.log 2>&1 && echo ok1
python figures/export_phase4_figures.py >/tmp/e2.log 2>&1 && echo ok2
python figures/export_phase4_full_suite_figures.py >/tmp/e3.log 2>&1 && echo ok3
ls thesis/figures/*.pdf | wc -l
```
Expected: `ok1 ok2 ok3` and a non-zero count of figure PDFs in `thesis/figures/`.

- [ ] **Step 7: Confirm clean status**

Run: `git status --porcelain`
Expected: empty (everything committed). Untracked large dirs (`results_*`, `results_archive/`) are gitignored and should not appear.

---

## Done

The repo now has five clear top-level categories (`experiments/` + `experiments/run/`,
`analysis/`, `figures/`, `results_*` / `results_archive/`, `thesis/`), a README + runbook
pipeline, `requirements.txt`, and `.env.example`. No experiment data was deleted; the
three unused results dirs are archived. Merge `chore/repo-reorg` when satisfied.
````
