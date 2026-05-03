# Stage 4 Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce all figures, tables, and supporting data needed to write `ch5_results.tex` §5.4 (Stage 4 results) per the design at `docs/superpowers/specs/2026-05-03-stage4-analysis-design.md`.

**Architecture:** Two layers. (1) A small reusable analysis package under `analysis/phase4/` for code that needs unit testing — failure-mode classifier, steering-success metric, code-identity extractor. (2) An updated working notebook `analysis/phase4_analysis.ipynb` that imports those modules and produces the section's figures + CSV tables under `analysis/figs_phase4/`. The §5.4.6 sub-experiment lives as a standalone runner under `experiments/` plus its own analysis notebook.

**Tech Stack:** Python 3.13, pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, IOHexperimenter, BLADE (`iohblade`), LLaMEA (`llamea.utils`). Existing conda env at `/opt/miniconda3/`. Tests run under `pytest`.

---

## File structure

**Create:**

- `analysis/phase4/__init__.py` — package marker.
- `analysis/phase4/failure_modes.py` — re-runs failed candidate code through BLADE's compile + smoke-test pipeline and labels each failure with one of `{code_generation, interface_mismatch, runtime_error, import_violation}`.
- `analysis/phase4/steering.py` — computes per-(condition, feature) "% candidates moved toward Stage 1 top-10 % reference" relative to vanilla.
- `analysis/phase4/code_identity.py` — pulls the winning algorithm's source for each condition; computes code metrics (LoC, max nesting depth); supplies a structured prompt template for LLM-assisted family classification (manual cross-check expected).
- `tests/analysis/test_failure_modes.py`, `tests/analysis/test_steering.py`, `tests/analysis/test_code_identity.py` — unit tests for the three modules.
- `experiments/phase4_full_suite_config.py` — config for the §5.4.6 sub-experiment (4 winners + CMA-ES; 1 000 functions × 5 instances × {5, 10, 20} dims; 2 000d budget; no LLM in loop).
- `experiments/phase4_full_suite_runner.py` — runs a single (algorithm, dimension) shard against the MA-BBOB pool and writes per-instance AOCCs to a parquet shard under `results_phase4_full_suite/`.
- `run_phase4_full_suite.py` — top-level driver mirroring `run_phase4.py`'s style; iterates over the 5 algorithms × 3 dimensions and dispatches shards.
- `slurm/phase4_full_suite.sh` — SLURM submission template for saronite (or another LIACS node).
- `analysis/phase4_full_suite.ipynb` — analyses the §5.4.6 results.

**Modify:**

- `analysis/phase4_analysis.ipynb` — extend with new cells that use the modules above; replace `±std` ribbons with 95 % bootstrap CI / SEM (revision-plan note 34); add an explicit ch5-mapped subsection ordering; emit one CSV per table.

**Out of scope:** modifying any existing experiment config, BLADE, or LLaMEA submodule code. The user's `feedback_dont_modify_existing.md` rule applies.

---

## Conventions used by every task

- **Tests use pytest fixtures** that point at a tiny in-repo sample under `tests/analysis/fixtures/` so we never depend on the full `results_phase4/` for unit tests. Where a fixture is needed, the task says exactly which file to create.
- **Figures** are written to `analysis/figs_phase4/` as PDF. Sister CSVs holding the underlying numbers are written to the same directory so LaTeX can `\input` them or so the table can be regenerated.
- **Commits** never use `--amend`, never include the `Co-Authored-By` trailer (per memory `feedback_no_coauthor.md`), and group per task. Run pre-commit / lint tools that already exist; do not add new ones.

---

## Phase A — Verify existing §5.4.1 / §5.4.2 cells regenerate cleanly

Stage 4 already has working aggregate-AOCC and per-instance figures in the existing notebook. We just need to confirm they re-execute against the current data and that nothing is silently stale. We also handle revision-plan note 34 (replace ±std with SEM / 95 % CI on convergence ribbons).

### Task A1: Re-execute the existing notebook end-to-end and confirm published figures

**Files:**
- Read: `analysis/phase4_analysis.ipynb`
- Modify: `analysis/figs_phase4/p4_*.pdf` (regenerated)

- [ ] **Step 1: Execute the notebook with errors allowed**

```bash
cd /Users/maxharell/repos/thesis
jupyter nbconvert --to notebook --execute analysis/phase4_analysis.ipynb \
  --output /tmp/phase4_executed.ipynb \
  --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.allow_errors=True
```

Expected: completes with two known errors in the t-SNE cells (cell 25, 26 — `np.nanquantile` shape; out of scope).

- [ ] **Step 2: Verify the regenerated PDFs are present and non-empty**

```bash
ls -la analysis/figs_phase4/p4_failure_rates.pdf analysis/figs_phase4/p4_final_aocc_boxplot.pdf analysis/figs_phase4/p4_convergence.pdf analysis/figs_phase4/p4_per_instance_heatmap.pdf analysis/figs_phase4/p4_behavioural_profiles.pdf analysis/figs_phase4/p4_budget_to_threshold.pdf
```

Expected: each file > 5 kB and modified within the last minute.

- [ ] **Step 3: Open each PDF visually**

```bash
open analysis/figs_phase4/p4_final_aocc_boxplot.pdf
open analysis/figs_phase4/p4_per_instance_heatmap.pdf
open analysis/figs_phase4/p4_convergence.pdf
```

Confirm the four conditions appear in the order `vanilla → neutral → sage → combined_neutral`, the per-instance heatmap shows 4 × 20, and the convergence curves separate around gen 100. If any figure looks wrong, stop and triage; do not commit.

- [ ] **Step 4: Commit any regenerated PDFs only if visually confirmed**

```bash
git add analysis/figs_phase4/p4_failure_rates.pdf analysis/figs_phase4/p4_final_aocc_boxplot.pdf analysis/figs_phase4/p4_convergence.pdf analysis/figs_phase4/p4_per_instance_heatmap.pdf analysis/figs_phase4/p4_behavioural_profiles.pdf analysis/figs_phase4/p4_budget_to_threshold.pdf
git commit -m "Regenerate Stage 4 §5.4.1/§5.4.2 figures from current data"
```

### Task A2: Replace ±std ribbon with 95 % bootstrap CI on the convergence figure

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (the cell that produces `p4_convergence.pdf`, currently around line 220 of the script export)

- [ ] **Step 1: Locate the convergence-curve cell**

Open `analysis/phase4_analysis.ipynb` in Jupyter and find the cell containing:

```python
ax.fill_between(sub.generation, sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                color=COND_COLORS[c], alpha=0.15)
```

- [ ] **Step 2: Replace the ribbon with a per-generation bootstrap CI**

Replace the `curves = (df.groupby(...))` block plus the plotting block with:

```python
def boot_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])

curves = []
for c in CONDITIONS:
    sub = df[df.condition == c]
    for g, gd in sub.groupby('generation'):
        vals = gd['best_so_far'].dropna().values
        if len(vals) < 2:
            continue
        lo, hi = boot_ci(vals)
        curves.append({'condition': c, 'generation': g,
                       'mean': vals.mean(), 'ci_lo': lo, 'ci_hi': hi})
curves = pd.DataFrame(curves)

fig, ax = plt.subplots(figsize=(7, 4))
for c in CONDITIONS:
    sub = curves[curves.condition == c]
    ax.plot(sub.generation, sub['mean'], label=COND_LABELS[c],
            color=COND_COLORS[c], lw=1.8)
    ax.fill_between(sub.generation, sub['ci_lo'], sub['ci_hi'],
                    color=COND_COLORS[c], alpha=0.15)
ax.set_xlabel('Generation (candidate evaluation)')
ax.set_ylabel('Best-so-far AOCC (mean, 95% bootstrap CI)')
ax.set_title('Convergence dynamics')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_convergence.pdf', bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Re-execute that cell only and visually verify**

In Jupyter: `Cell → Run`. Then:

```bash
open analysis/figs_phase4/p4_convergence.pdf
```

Expected: ribbons are tighter than the previous ±std bands, especially for `combined`. Curves are unchanged.

- [ ] **Step 4: Commit**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_convergence.pdf
git commit -m "Use 95% bootstrap CI on Stage 4 convergence ribbon (revision note 34)"
```

---

## Phase B — §5.4.3 Failure-rate analysis

The novel work in this section is failure-mode classification. Phase 4 didn't store a per-failure error string in the logs (verified: `error` field is empty). However, the failed candidates' source code *is* preserved. We re-run the same compile + smoke-test pipeline BLADE used to produce labels post-hoc.

### Task B1: Scaffold the analysis package

**Files:**
- Create: `analysis/phase4/__init__.py`
- Create: `tests/analysis/__init__.py`
- Create: `tests/analysis/conftest.py`
- Create: `tests/analysis/fixtures/.gitkeep`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p analysis/phase4 tests/analysis/fixtures
touch analysis/phase4/__init__.py tests/analysis/__init__.py tests/analysis/fixtures/.gitkeep
```

- [ ] **Step 2: Create a conftest with a tiny shared fixture**

`tests/analysis/conftest.py`:

```python
"""Shared fixtures for analysis-package tests."""
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 3: Verify pytest discovers the empty package**

```bash
cd /Users/maxharell/repos/thesis
pytest tests/analysis -q
```

Expected: `no tests ran in 0.0Xs` (success, just nothing to do).

- [ ] **Step 4: Commit**

```bash
git add analysis/phase4/__init__.py tests/analysis/__init__.py tests/analysis/conftest.py tests/analysis/fixtures/.gitkeep
git commit -m "Scaffold analysis/phase4 package and tests/analysis fixtures dir"
```

### Task B2: Write failing tests for the failure-mode classifier

**Files:**
- Create: `tests/analysis/fixtures/code_syntax_error.py`
- Create: `tests/analysis/fixtures/code_no_class.py`
- Create: `tests/analysis/fixtures/code_bad_init.py`
- Create: `tests/analysis/fixtures/code_bad_call.py`
- Create: `tests/analysis/fixtures/code_runtime_error.py`
- Create: `tests/analysis/fixtures/code_valid.py`
- Create: `tests/analysis/fixtures/code_disallowed_import.py`
- Create: `tests/analysis/test_failure_modes.py`

- [ ] **Step 1: Create one fixture file per failure category**

`tests/analysis/fixtures/code_syntax_error.py`:

```python
import numpy as np

class Broken
    def __init__(self, budget, dim):
        pass
```

`tests/analysis/fixtures/code_no_class.py`:

```python
import numpy as np

def some_function(budget, dim):
    return budget * dim
```

`tests/analysis/fixtures/code_bad_init.py`:

```python
import numpy as np

class BadInit:
    def __init__(self):
        self.budget = 100

    def __call__(self, func):
        for _ in range(self.budget):
            func(np.zeros(2))
```

`tests/analysis/fixtures/code_bad_call.py`:

```python
import numpy as np

class BadCall:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self):
        return None
```

`tests/analysis/fixtures/code_runtime_error.py`:

```python
import numpy as np

class RuntimeBoom:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        return 1 / 0
```

`tests/analysis/fixtures/code_valid.py`:

```python
import numpy as np

class ValidRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            func(x)
```

`tests/analysis/fixtures/code_disallowed_import.py`:

```python
import torch

class TorchDep:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pass
```

- [ ] **Step 2: Write the test file**

`tests/analysis/test_failure_modes.py`:

```python
"""Unit tests for analysis.phase4.failure_modes."""
from pathlib import Path

import pytest

from analysis.phase4.failure_modes import classify_failure


def _read(fixture_dir: Path, name: str) -> str:
    return (fixture_dir / name).read_text()


@pytest.mark.parametrize('fname,expected', [
    ('code_syntax_error.py', 'code_generation'),
    ('code_no_class.py', 'code_generation'),
    ('code_bad_init.py', 'interface_mismatch'),
    ('code_bad_call.py', 'interface_mismatch'),
    ('code_runtime_error.py', 'runtime_error'),
    ('code_disallowed_import.py', 'import_violation'),
])
def test_classify_known_failures(fixture_dir: Path, fname: str, expected: str):
    code = _read(fixture_dir, fname)
    label, _detail = classify_failure(code)
    assert label == expected


def test_valid_code_returns_none(fixture_dir: Path):
    """Code that passes the compile + smoke pipeline classifies as None."""
    code = _read(fixture_dir, 'code_valid.py')
    label, _ = classify_failure(code)
    assert label is None


def test_classify_returns_detail_string(fixture_dir: Path):
    code = _read(fixture_dir, 'code_runtime_error.py')
    label, detail = classify_failure(code)
    assert label == 'runtime_error'
    assert 'division' in detail.lower() or 'zero' in detail.lower()
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
pytest tests/analysis/test_failure_modes.py -v
```

Expected: every test errors with `ModuleNotFoundError: No module named 'analysis.phase4.failure_modes'`.

- [ ] **Step 4: Commit fixtures + tests only**

```bash
git add tests/analysis/fixtures/*.py tests/analysis/test_failure_modes.py
git commit -m "Add failing tests for Stage 4 failure-mode classifier"
```

### Task B3: Implement the failure-mode classifier

**Files:**
- Create: `analysis/phase4/failure_modes.py`

- [ ] **Step 1: Implement the module**

`analysis/phase4/failure_modes.py`:

```python
"""Re-classify Stage 4 failures by re-running the BLADE compile + smoke pipeline.

The original Stage 4 logs do not preserve the per-failure error string. We
recover the failure category by replaying the candidate's source code through
the same gate the framework used at runtime.

Categories:
  - code_generation:    no class is produced or the file does not parse.
  - import_violation:   the code imports a disallowed package.
  - interface_mismatch: __init__(budget, dim) or __call__(func) cannot be
                        invoked with the framework's call signature.
  - runtime_error:      compiles and instantiates, but raises during the
                        first BBOB-level call.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Tuple

_THESIS_ROOT = Path(__file__).resolve().parents[2]


def _load_llamea_utils():
    spec = importlib.util.spec_from_file_location(
        'llamea.utils',
        os.path.join(str(_THESIS_ROOT), 'LLaMEA', 'llamea', 'utils.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ALLOWED_IMPORTS = ('numpy',)  # mirrors experiments/phase1_config.py:ALLOWED_IMPORTS


def classify_failure(code: str) -> Tuple[str | None, str]:
    """Return (label, detail). label is None for code that runs cleanly.

    The smoke test instantiates the candidate with budget=100, dim=2 and
    invokes it on BBOB function 11 (Discus), instance 1, dim 2.
    """
    if not code or 'class ' not in code:
        return 'code_generation', 'no class definition found'

    llamea_utils = _load_llamea_utils()

    # Stage 1 — compile.
    global_ns = {'__name__': '__main__'}
    local_ns: dict = {}
    try:
        global_ns, _ = llamea_utils.prepare_namespace(
            code, allowed=_ALLOWED_IMPORTS,
        )
    except ImportError as e:
        return 'import_violation', str(e)
    except SyntaxError as e:
        return 'code_generation', str(e)

    try:
        exec(code, global_ns, local_ns)
    except SyntaxError as e:
        return 'code_generation', str(e)
    except ImportError as e:
        return 'import_violation', str(e)
    except Exception as e:
        return 'code_generation', f'{type(e).__name__}: {e}'

    from iohblade.utils import first_class_name
    cls_name = first_class_name(code)
    if cls_name is None or cls_name not in local_ns:
        return 'code_generation', 'class not found in compiled namespace'

    # Stage 2 — smoke test.
    try:
        import ioh
        from ioh import logger as ioh_logger
        from iohblade.utils import aoc_logger, OverBudgetException
    except ImportError as e:
        # The host environment is broken; surface to caller rather than mislabel.
        raise RuntimeError(f'cannot import ioh/iohblade for smoke test: {e}') from e

    try:
        algo = local_ns[cls_name](budget=100, dim=2)
    except TypeError as e:
        return 'interface_mismatch', f'__init__ failed: {e}'
    except Exception as e:
        return 'runtime_error', f'__init__ raised {type(e).__name__}: {e}'

    try:
        l_tmp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        prob = ioh.get_problem(11, 1, 2)
        prob.attach_logger(l_tmp)
        algo(prob)
    except OverBudgetException:
        return None, 'ok'
    except TypeError as e:
        return 'interface_mismatch', f'__call__ failed: {e}'
    except Exception as e:
        return 'runtime_error', f'{type(e).__name__}: {e}'

    return None, 'ok'
```

- [ ] **Step 2: Run the tests to confirm they pass**

```bash
pytest tests/analysis/test_failure_modes.py -v
```

Expected: 8 / 8 PASS. If `code_disallowed_import` does not pick up `import_violation`, `prepare_namespace` may behave differently than expected — open `LLaMEA/llamea/utils.py` and align the test to actual behaviour.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4/failure_modes.py
git commit -m "Implement Stage 4 failure-mode classifier"
```

### Task B4: Apply classifier to all Stage 4 failures and persist labels

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (new cell)
- Create: `analysis/figs_phase4/p4_failure_modes.csv`

- [ ] **Step 1: Add a cell that walks every condition × seed**

Append a new cell at the end of the notebook (under a new markdown header `## §5.4.3 Failure-mode classification`):

```python
import json
from pathlib import Path

from analysis.phase4.failure_modes import classify_failure

records = []
for cond in CONDITIONS:
    for seed in SEEDS:
        run_glob = list((RESULTS_DIR / cond / f'seed-{seed}').glob('run-*/log.jsonl'))
        if not run_glob:
            continue
        with open(run_glob[0]) as f:
            for line in f:
                row = json.loads(line)
                if row.get('fitness') in (None, '-inf') or row.get('fitness') == float('-inf'):
                    label, detail = classify_failure(row.get('code', ''))
                    records.append({
                        'condition': cond,
                        'seed': seed,
                        'generation': row.get('generation'),
                        'name': row.get('name'),
                        'label': label or 'unclassified_success',
                        'detail': detail,
                    })

fail_df = pd.DataFrame(records)
fail_df.to_csv(FIG_DIR / 'p4_failure_modes.csv', index=False)
print(f'classified {len(fail_df)} failed candidates')
print(fail_df.groupby(['condition', 'label']).size().unstack(fill_value=0).loc[CONDITIONS])
```

- [ ] **Step 2: Execute the cell and skim the breakdown**

Expected: ~3 462 total failures across the four conditions (vanilla 839, neutral 696, sage 1 048, combined 879). The breakdown table prints with non-zero counts in `code_generation`, `interface_mismatch`, and `runtime_error`. If `unclassified_success` appears, the upstream csv `run_status` and the per-row `fitness` disagree — investigate.

- [ ] **Step 3: Commit the cell + CSV**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_failure_modes.csv
git commit -m "Classify all Stage 4 failures and persist label table"
```

### Task B5: Failure-mode breakdown table

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (new cell)
- Create: `analysis/figs_phase4/p4_failure_mode_breakdown.csv`

- [ ] **Step 1: Add a table-formatting cell**

```python
breakdown = (
    fail_df.groupby(['condition', 'label']).size()
    .unstack(fill_value=0)
    .loc[CONDITIONS]
)
breakdown['total'] = breakdown.sum(axis=1)
for col in [c for c in breakdown.columns if c != 'total']:
    breakdown[f'{col}_pct'] = (100 * breakdown[col] / breakdown['total']).round(1)
breakdown.to_csv(FIG_DIR / 'p4_failure_mode_breakdown.csv')
print(breakdown)
```

- [ ] **Step 2: Execute and visually verify**

Expected columns: `code_generation`, `interface_mismatch`, `runtime_error`, plus `_pct` variants and `total`. The percentages within each row should sum to ~100.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_failure_mode_breakdown.csv
git commit -m "Add Stage 4 failure-mode breakdown table"
```

### Task B6: Generation-binned failure-rate figure

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (new cell)
- Create: `analysis/figs_phase4/p4_failure_rate_by_gen.pdf`

- [ ] **Step 1: Add a cell that bins failures by generation**

```python
bins = [0, 100, 200, 300, 400, 500]
labels = ['0-99', '100-199', '200-299', '300-399', '400-499']
df['gen_bin'] = pd.cut(df['generation'], bins=bins, labels=labels,
                       right=False, include_lowest=True)

bin_rate = (
    df.groupby(['condition', 'gen_bin'], observed=True)
    .agg(n=('run_status', 'size'),
         fail=('run_status', lambda s: (s == 'failure').sum()))
    .assign(rate=lambda x: 100 * x['fail'] / x['n'])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7.5, 4))
for c in CONDITIONS:
    sub = bin_rate[bin_rate.condition == c]
    ax.plot(sub.gen_bin, sub.rate, marker='o', lw=1.8,
            color=COND_COLORS[c], label=COND_LABELS[c])
ax.set_xlabel('Generation bin')
ax.set_ylabel('Failure rate (%)')
ax.set_title('Failure rate by 100-candidate generation bin')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_failure_rate_by_gen.pdf', bbox_inches='tight')
plt.show()
bin_rate.to_csv(FIG_DIR / 'p4_failure_rate_by_gen.csv', index=False)
```

- [ ] **Step 2: Execute and visually verify**

Open the PDF. Expected: x-axis with 5 bins; four curves; sage typically tracks above the others. If the curves are flat, the brittle-code-accumulates hypothesis is unsupported and we say so in prose.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_failure_rate_by_gen.pdf analysis/figs_phase4/p4_failure_rate_by_gen.csv
git commit -m "Add Stage 4 failure-rate-by-generation figure"
```

---

## Phase C — §5.4.4 Steering-success quantification

The notebook already produces the per-condition feature-medians and the violins. We add the missing piece: a per-(condition, feature) "% candidates that moved toward the Stage 1 top-10 % reference relative to vanilla" metric (revision-plan note 9).

### Task C1: Failing tests for steering metric

**Files:**
- Create: `tests/analysis/test_steering.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for analysis.phase4.steering."""
import numpy as np
import pandas as pd
import pytest

from analysis.phase4.steering import steering_rate


def _frame(values, condition, feature='intensification_ratio'):
    return pd.DataFrame({
        'condition': condition,
        f'bm_{feature}': values,
    })


def test_steers_toward_higher_reference():
    """When advised direction is 'up' and condition pushes feature higher than
    vanilla, steering rate should exceed 50%."""
    vanilla = _frame(np.full(100, 0.5), 'vanilla')
    cond = _frame(np.full(100, 0.7), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    assert rate == pytest.approx(100.0)


def test_steers_against_higher_reference():
    """If the condition's median is below vanilla's, steering rate should be 0."""
    vanilla = _frame(np.full(100, 0.7), 'vanilla')
    cond = _frame(np.full(100, 0.5), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    assert rate == pytest.approx(0.0)


def test_handles_lower_is_better():
    vanilla = _frame(np.full(100, 0.7), 'vanilla')
    cond = _frame(np.full(100, 0.3), 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='down')
    assert rate == pytest.approx(100.0)


def test_drops_nans():
    vanilla = _frame(np.full(100, 0.5), 'vanilla')
    cond = _frame([0.7, np.nan, 0.7] * 33 + [0.7], 'neutral')
    df = pd.concat([vanilla, cond])
    rate = steering_rate(df, feature='intensification_ratio',
                         condition='neutral', vanilla='vanilla',
                         direction='up')
    # All non-NaN values are above vanilla's 0.5 → 100 %
    assert rate == pytest.approx(100.0)


def test_invalid_direction_raises():
    df = _frame([0.5, 0.6], 'neutral')
    with pytest.raises(ValueError):
        steering_rate(df, feature='intensification_ratio',
                      condition='neutral', vanilla='vanilla',
                      direction='sideways')
```

- [ ] **Step 2: Run and confirm import errors**

```bash
pytest tests/analysis/test_steering.py -v
```

Expected: 5 ImportErrors.

- [ ] **Step 3: Commit**

```bash
git add tests/analysis/test_steering.py
git commit -m "Add failing tests for steering-success metric"
```

### Task C2: Implement the steering metric

**Files:**
- Create: `analysis/phase4/steering.py`

- [ ] **Step 1: Implement the module**

```python
"""Quantify how much each Stage 4 condition steered behaviour relative to vanilla.

`steering_rate` returns the percentage of valid candidates whose feature value
moved in the *advised* direction relative to vanilla's median. The advised
direction is taken from the Stage 1 Spearman analysis and supplied by the
caller — neutral feedback uses no explicit direction, so we measure whether
the implicit signal still lands.
"""
from __future__ import annotations

import pandas as pd

_DIRECTIONS = ('up', 'down')


def steering_rate(df: pd.DataFrame, *, feature: str,
                  condition: str, vanilla: str,
                  direction: str) -> float:
    """Return the percentage of `condition`'s candidates whose `feature` value
    is on the advised side of vanilla's median.

    df must have columns 'condition' and f'bm_{feature}'.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f'direction must be one of {_DIRECTIONS}, got {direction!r}')

    col = f'bm_{feature}'
    vanilla_median = df.loc[df['condition'] == vanilla, col].dropna().median()
    cand = df.loc[df['condition'] == condition, col].dropna()
    if len(cand) == 0:
        return float('nan')

    if direction == 'up':
        moved = (cand > vanilla_median).sum()
    else:
        moved = (cand < vanilla_median).sum()
    return 100.0 * moved / len(cand)
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/analysis/test_steering.py -v
```

Expected: 5 / 5 PASS.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4/steering.py
git commit -m "Implement Stage 4 steering-success metric"
```

### Task C3: Apply steering metric across the five tracked features and emit table

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (new cell)
- Create: `analysis/figs_phase4/p4_steering_rates.csv`

- [ ] **Step 1: Add the application cell**

```python
from analysis.phase4.steering import steering_rate

# Direction comes from the Stage 1 Spearman analysis (ch5 §5.2.5 table).
# Higher is better unless noted.
ADVISED = {
    'intensification_ratio': 'up',
    'dimension_convergence_heterogeneity': 'up',
    'fitness_plateau_fraction': 'up',
    'avg_improvement': 'down',
    'improvement_spatial_correlation': 'up',
}

rows = []
for feat, direction in ADVISED.items():
    for c in [c for c in CONDITIONS if c != 'vanilla']:
        rate = steering_rate(valid, feature=feat,
                             condition=c, vanilla='vanilla',
                             direction=direction)
        rows.append({'feature': feat, 'condition': c,
                     'direction': direction, 'rate_pct': round(rate, 1)})
steering_df = pd.DataFrame(rows).pivot(
    index='feature', columns='condition', values='rate_pct'
).loc[list(ADVISED.keys()), [c for c in CONDITIONS if c != 'vanilla']]
steering_df.to_csv(FIG_DIR / 'p4_steering_rates.csv')
print(steering_df)
```

- [ ] **Step 2: Execute and skim**

Expected: rows for each tracked feature, columns for `neutral / sage / combined_neutral`. Values in 0–100. The hypothesis is that combined achieves the highest steering rate across most features (consistent with its "stereotypically high-AOCC" profile).

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_steering_rates.csv
git commit -m "Add Stage 4 steering-success rates per feature × condition"
```

---

## Phase D — §5.4.5 Four-best-algorithm code identity

For each condition, the seed whose final best-so-far AOCC is highest produces a "winner". We extract their source, compute light code metrics, and produce a structured prompt for an LLM-assisted family classification (which is then sanity-checked manually before ending up in the thesis table).

### Task D1: Failing tests for code-identity helpers

**Files:**
- Create: `tests/analysis/test_code_identity.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for analysis.phase4.code_identity."""
import textwrap

from analysis.phase4.code_identity import (
    code_metrics,
    family_prompt,
)


def test_code_metrics_counts_loc_and_depth():
    code = textwrap.dedent('''
        import numpy as np

        class Demo:
            def __init__(self, budget, dim):
                self.b = budget
                if budget > 0:
                    for i in range(3):
                        if i % 2 == 0:
                            print(i)
    ''').strip()
    m = code_metrics(code)
    assert m['lines_of_code'] >= 8
    assert m['max_nesting'] >= 4   # def → if → for → if


def test_code_metrics_skips_blank_and_comments():
    code = '\n'.join([
        '# top comment',
        '',
        'import numpy as np',
        '',
        'class A:',
        '    pass',
    ])
    m = code_metrics(code)
    assert m['lines_of_code'] == 3   # import, class, pass
    assert m['max_nesting'] == 1


def test_family_prompt_contains_class_name_and_code():
    code = '''
    import numpy as np

    class CMA_ES_Like:
        pass
    '''.strip()
    prompt = family_prompt(condition='neutral', algorithm_name='CMA_ES_Like',
                           code=code)
    assert 'CMA_ES_Like' in prompt
    assert 'neutral' in prompt
    assert 'CMA-ES' in prompt or 'metaheuristic family' in prompt
```

- [ ] **Step 2: Confirm they fail**

```bash
pytest tests/analysis/test_code_identity.py -v
```

Expected: 3 ImportErrors.

- [ ] **Step 3: Commit**

```bash
git add tests/analysis/test_code_identity.py
git commit -m "Add failing tests for Stage 4 code-identity helpers"
```

### Task D2: Implement the code-identity module

**Files:**
- Create: `analysis/phase4/code_identity.py`

- [ ] **Step 1: Implement**

```python
"""Helpers for §5.4.5: identify the four best-found algorithms.

`code_metrics` returns a small dict of static metrics. `family_prompt` returns
a single-string prompt designed to feed an LLM a structured task: classify the
metaheuristic family and list named components. The user is expected to
sanity-check the LLM's output by reading the code; this module never calls the
LLM itself.
"""
from __future__ import annotations

import textwrap


def _strip_comments_and_blank(code: str) -> list[str]:
    out: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        out.append(line)
    return out


def code_metrics(code: str) -> dict:
    """Return static code metrics: lines of code (sans blank/comment) and the
    deepest indentation level seen."""
    lines = _strip_comments_and_blank(code)
    max_indent = 0
    for line in lines:
        leading = len(line) - len(line.lstrip(' '))
        # Treat 4 spaces as one nesting level. Tabs are normalised to 4.
        leading = leading + line.count('\t', 0, leading) * 3
        max_indent = max(max_indent, leading // 4)
    return {
        'lines_of_code': len(lines),
        'max_nesting': max_indent + 1,  # +1 so a top-level statement reads as 1, not 0
    }


def family_prompt(*, condition: str, algorithm_name: str, code: str) -> str:
    """Build the LLM prompt that asks for a metaheuristic-family verdict."""
    return textwrap.dedent(f'''
        You are reading the source code of a metaheuristic optimisation
        algorithm generated by an LLM. The algorithm is named
        {algorithm_name} and was produced under feedback condition
        '{condition}' in a {{(1+1)}}-ES evolutionary loop on the MA-BBOB
        benchmark.

        Classify the algorithm's metaheuristic family. Pick exactly one of:
          - CMA-ES variant
          - Differential Evolution
          - Particle Swarm Optimisation
          - Nelder-Mead / Simplex
          - Bayesian / surrogate
          - Hybrid / composite
          - Novel (does not match a known family)

        Then list, in bullet form, the named components present (e.g. covariance
        adaptation, restart strategy, step-size control, archive). Quote the
        exact identifier from the code.

        Source code:

        ```python
        {code}
        ```

        Respond with two sections: 'Family:' on one line, then 'Components:' as
        a bullet list.
    ''').strip()
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/analysis/test_code_identity.py -v
```

Expected: 3 / 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4/code_identity.py
git commit -m "Implement Stage 4 code-identity helpers"
```

### Task D3: Extract the four winners and write code metrics + family prompts

**Files:**
- Modify: `analysis/phase4_analysis.ipynb` (new cell)
- Create: `analysis/figs_phase4/p4_winners/{vanilla,neutral,sage,combined_neutral}_winner.py`
- Create: `analysis/figs_phase4/p4_winners/{vanilla,neutral,sage,combined_neutral}_prompt.txt`
- Create: `analysis/figs_phase4/p4_winner_metrics.csv`

- [ ] **Step 1: Add the extraction cell**

```python
import json
from pathlib import Path

from analysis.phase4.code_identity import code_metrics, family_prompt

WINNERS_DIR = FIG_DIR / 'p4_winners'
WINNERS_DIR.mkdir(parents=True, exist_ok=True)

# Best-by-final-AOCC seed per condition.
best_seed = (final.loc[final.groupby('condition')['final_best'].idxmax()]
             [['condition', 'seed', 'final_best']]
             .set_index('condition'))

records = []
for cond in CONDITIONS:
    seed = int(best_seed.loc[cond, 'seed'])
    aocc = float(best_seed.loc[cond, 'final_best'])
    log_path = next((RESULTS_DIR / cond / f'seed-{seed}').glob('run-*/log.jsonl'))
    with open(log_path) as f:
        rows = [json.loads(l) for l in f]
    valid_rows = [r for r in rows
                  if isinstance(r.get('fitness'), (int, float))
                  and r['fitness'] != float('-inf')]
    winner = max(valid_rows, key=lambda r: r['fitness'])

    code = winner['code']
    name = winner['name']
    (WINNERS_DIR / f'{cond}_winner.py').write_text(
        f'# condition: {cond}\n# seed: {seed}\n# AOCC: {aocc:.4f}\n# name: {name}\n\n'
        + code
    )
    (WINNERS_DIR / f'{cond}_prompt.txt').write_text(
        family_prompt(condition=cond, algorithm_name=name, code=code)
    )

    metrics = code_metrics(code)
    records.append({
        'condition': cond,
        'seed': seed,
        'algorithm_name': name,
        'final_aocc': round(aocc, 4),
        **metrics,
    })

metrics_df = pd.DataFrame(records).set_index('condition').loc[CONDITIONS]
metrics_df.to_csv(FIG_DIR / 'p4_winner_metrics.csv')
print(metrics_df)
```

- [ ] **Step 2: Execute and verify the four `*_winner.py` and `*_prompt.txt` files exist**

```bash
ls analysis/figs_phase4/p4_winners/
cat analysis/figs_phase4/p4_winners/sage_winner.py | head -20
```

Expected: 8 files (4 winners × 2 each), plus the metrics CSV. Each `*_winner.py` opens with the condition / seed / AOCC / name header.

- [ ] **Step 3: Commit**

```bash
git add analysis/phase4_analysis.ipynb analysis/figs_phase4/p4_winners/ analysis/figs_phase4/p4_winner_metrics.csv
git commit -m "Extract Stage 4 winning algorithms with code metrics and identity prompts"
```

### Task D4: Manual family classification and final identity table

**Files:**
- Create: `analysis/figs_phase4/p4_winner_identity.csv`

- [ ] **Step 1: Run the four prompts through Claude (or any capable LLM) by hand**

For each `analysis/figs_phase4/p4_winners/<cond>_prompt.txt`:

1. Open the prompt in your editor (VS Code).
2. Copy the contents.
3. Paste into a Claude conversation and submit.
4. Save Claude's response to `analysis/figs_phase4/p4_winners/<cond>_response.txt`.

Skim each response. Cross-check the family verdict by reading 30-50 lines of the corresponding `<cond>_winner.py`. Don't accept a verdict you can't trace to specific identifiers in the code.

- [ ] **Step 2: Hand-write the merged table**

`analysis/figs_phase4/p4_winner_identity.csv`:

```csv
condition,seed,algorithm_name,family,components,distinctive_feature,final_aocc
vanilla,<seed>,<name>,<family>,"<comma-separated components>",<one-line distinctive feature>,<aocc>
neutral,<seed>,<name>,<family>,"<comma-separated components>",<one-line distinctive feature>,<aocc>
sage,<seed>,<name>,<family>,"<comma-separated components>",<one-line distinctive feature>,<aocc>
combined_neutral,<seed>,<name>,<family>,"<comma-separated components>",<one-line distinctive feature>,<aocc>
```

Fill in the values from your manual review and the saved metrics CSV. Keep the `family` field to one of the seven categories the prompt enumerates — if none fit, use `Novel` and explain in `distinctive_feature`.

- [ ] **Step 3: Commit**

```bash
git add analysis/figs_phase4/p4_winners/*_response.txt analysis/figs_phase4/p4_winner_identity.csv
git commit -m "Classify Stage 4 winners by metaheuristic family"
```

---

## Phase E — §5.4.6 Full-suite generalisation experiment

The four winners (from Phase D) plus CMA-ES are evaluated on all 1 000 MA-BBOB functions × 5 instances × {5, 10, 20} dimensions under a 2 000d FE budget. No LLM in the loop. Total ≈ 1.75 × 10⁹ FEs across the five algorithms.

### Task E1: Experiment config

**Files:**
- Create: `experiments/phase4_full_suite_config.py`

- [ ] **Step 1: Author the config**

```python
"""Phase 4 full-suite generalisation: 4 LLM winners + CMA-ES on all 1 000
MA-BBOB functions × 5 instances × {5, 10, 20} dims under 2 000d budget.

Per-feedback-rule (`feedback_dont_modify_existing`), this is a NEW config
file rather than an edit of `phase4_config.py`.
"""
import os

from .phase1_config import BBOB_BOUNDS

EVAL_TIMEOUT = 1800  # 30 min per algorithm × dim × instance shard

# Test set
ALL_INSTANCES = list(range(1000))
EVAL_SEEDS = 5

# Dimensions and budget
DIMS = [5, 10, 20]
BUDGET_FACTOR = 2000  # FEs = 2000 * dim, BBOB convention

# Algorithm shards
ALGORITHMS = {
    'vanilla_winner':          'analysis/figs_phase4/p4_winners/vanilla_winner.py',
    'neutral_winner':          'analysis/figs_phase4/p4_winners/neutral_winner.py',
    'sage_winner':             'analysis/figs_phase4/p4_winners/sage_winner.py',
    'combined_neutral_winner': 'analysis/figs_phase4/p4_winners/combined_neutral_winner.py',
    'cma_es':                  'BUILTIN:cma_es',  # served by the runner directly
}

RESULTS_DIR = os.environ.get('PHASE4_FULL_SUITE_DIR', 'results_phase4_full_suite')
```

- [ ] **Step 2: Sanity import**

```bash
python -c "from experiments import phase4_full_suite_config as c; print(len(c.ALL_INSTANCES), c.DIMS, list(c.ALGORITHMS))"
```

Expected: `1000 [5, 10, 20] ['vanilla_winner', 'neutral_winner', 'sage_winner', 'combined_neutral_winner', 'cma_es']`.

- [ ] **Step 3: Commit**

```bash
git add experiments/phase4_full_suite_config.py
git commit -m "Add Stage 4.6 full-suite experiment config"
```

### Task E2: Runner module

**Files:**
- Create: `experiments/phase4_full_suite_runner.py`

- [ ] **Step 1: Implement the runner**

```python
"""Run a single (algorithm, dim, instance-batch) shard of the §5.4.6 sweep.

The runner deliberately uses no LLM. It compiles the saved winner code,
instantiates it, and evaluates each instance with the BBOB convention. CMA-ES
is supplied by `cma` (pycma) at the same budget for direct comparison.

Output: parquet under <RESULTS_DIR>/<alg>/<dim>/instances_<batch>.parquet.
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.phase4_full_suite_config import (
    ALGORITHMS,
    BBOB_BOUNDS,
    BUDGET_FACTOR,
    EVAL_SEEDS,
    EVAL_TIMEOUT,
    RESULTS_DIR,
)


def _load_user_algo(path: str):
    """exec the saved winner-code file and return the leaf class object."""
    src = Path(path).read_text()
    ns: dict = {}
    exec(src, ns, ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    if not classes:
        raise RuntimeError(f'no class defined in {path}')
    return classes[-1]   # the last-defined class is the algorithm


def _aocc(curve: np.ndarray, budget: int, lb: float = 1e-8, ub: float = 1e2) -> float:
    """Per-run AOCC, identical to thesis §4.5 eq. (4.4)."""
    log_curve = np.log10(np.clip(curve, lb, ub))
    log_lb, log_ub = np.log10(lb), np.log10(ub)
    return float(np.mean(1.0 - (log_curve - log_lb) / (log_ub - log_lb)))


_MA_BBOB_DATA = None  # cached MA_BBOB instance for instance-data lookups


def _ensure_ma_bbob_data():
    """Lazily instantiate MA_BBOB once to load self.weights, self.iids, self.opt_locs."""
    global _MA_BBOB_DATA
    if _MA_BBOB_DATA is None:
        from iohblade.benchmarks.BBOB.mabbob import MA_BBOB
        # We never call this instance — only read its weight/iid/opt_locs tables.
        _MA_BBOB_DATA = MA_BBOB(training_instances=[0], dims=[5], budget_factor=BUDGET_FACTOR)
    return _MA_BBOB_DATA


def _run_once(algo_factory, dim: int, instance_idx: int, seed: int) -> float:
    """Drive one (instance, seed) optimisation run and return its per-run AOCC."""
    import ioh  # imported lazily so unit tests don't require it

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Pull the MA-BBOB instance the same way as `MaBBOBProblem.evaluate`.
    data = _ensure_ma_bbob_data()
    f_new = ioh.problem.ManyAffine(
        xopt=np.array(data.opt_locs.iloc[instance_idx])[:dim],
        weights=np.array(data.weights.iloc[instance_idx]),
        instances=np.array(data.iids.iloc[instance_idx], dtype=int),
        n_variables=dim,
    )
    f_new.set_id(100)
    f_new.set_instance(instance_idx)

    budget = BUDGET_FACTOR * dim
    curve = np.full(budget, np.inf)
    pos = [0]

    def wrapped(x):
        if pos[0] >= budget:
            return 1e30
        y = float(f_new(x))
        curve[pos[0]] = min(y, curve[pos[0] - 1] if pos[0] > 0 else y)
        pos[0] += 1
        return y

    algo = algo_factory(budget=budget, dim=dim)
    try:
        algo(wrapped)
    except Exception:
        # Score the run with whatever curve we have so far.
        pass

    # Pad if the algorithm exited early.
    if pos[0] < budget:
        curve[pos[0]:] = curve[pos[0] - 1] if pos[0] > 0 else 1e2

    return _aocc(curve, budget)


def _cma_es_factory(budget: int, dim: int):
    import cma

    class _CMAESWrapper:
        def __init__(self, budget: int, dim: int):
            self._budget = budget
            self._dim = dim

        def __call__(self, func):
            es = cma.CMAEvolutionStrategy(
                np.zeros(self._dim), 1.0,
                {'bounds': [-5.0, 5.0], 'maxfevals': self._budget,
                 'verbose': -9},
            )
            while not es.stop():
                xs = es.ask()
                ys = [func(x) for x in xs]
                es.tell(xs, ys)
    return _CMAESWrapper(budget=budget, dim=dim)


def _factory(name: str):
    spec = ALGORITHMS[name]
    if spec.startswith('BUILTIN:cma_es'):
        return _cma_es_factory
    cls = _load_user_algo(spec)
    return cls


def run_shard(alg_name: str, dim: int, instance_indices: Iterable[int],
              out_dir: Path) -> Path:
    factory = _factory(alg_name)
    rows = []
    t0 = time.monotonic()
    for idx in instance_indices:
        for seed in range(EVAL_SEEDS):
            if time.monotonic() - t0 > EVAL_TIMEOUT:
                raise TimeoutError(f'shard {alg_name} dim={dim} timed out at instance={idx}')
            score = _run_once(factory, dim, idx, seed)
            rows.append({'algorithm': alg_name, 'dim': dim, 'instance': idx,
                         'eval_seed': seed, 'aocc': score})
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{alg_name}_dim{dim}_inst{min(instance_indices)}-{max(instance_indices)}.parquet'
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
```

- [ ] **Step 2: Sanity import**

```bash
python -c "from experiments import phase4_full_suite_runner as r; print(r.run_shard.__doc__ or 'ok')"
```

Expected: prints "ok" or the docstring.

- [ ] **Step 3: Commit**

```bash
git add experiments/phase4_full_suite_runner.py
git commit -m "Add Stage 4.6 full-suite runner module"
```

### Task E3: Smoke test the runner against one tiny shard

**Files:**
- Run-only (writes `results_phase4_full_suite/smoke/...`)

- [ ] **Step 1: Take the vanilla winner and run on instance 22 only, 5D**

(Phase D must have completed; `analysis/figs_phase4/p4_winners/vanilla_winner.py` exists.)

```bash
PHASE4_FULL_SUITE_DIR=results_phase4_full_suite/smoke \
python -c "
from pathlib import Path
from experiments.phase4_full_suite_runner import run_shard
out = run_shard('vanilla_winner', dim=5, instance_indices=[22], out_dir=Path('results_phase4_full_suite/smoke'))
print('wrote', out)
import pandas as pd
print(pd.read_parquet(out))
"
```

Expected: prints the parquet path and a 5-row dataframe (5 eval seeds × 1 instance) with finite AOCCs in [0, 1].

- [ ] **Step 2: Smoke-test CMA-ES the same way**

```bash
PHASE4_FULL_SUITE_DIR=results_phase4_full_suite/smoke \
python -c "
from pathlib import Path
from experiments.phase4_full_suite_runner import run_shard
out = run_shard('cma_es', dim=5, instance_indices=[22], out_dir=Path('results_phase4_full_suite/smoke'))
import pandas as pd
print(pd.read_parquet(out))
"
```

Expected: 5-row frame, finite AOCCs.

- [ ] **Step 3: Clean up the smoke output and commit nothing**

```bash
rm -rf results_phase4_full_suite/smoke
```

The smoke run is throwaway. No commit.

### Task E4: Top-level driver

**Files:**
- Create: `run_phase4_full_suite.py`

- [ ] **Step 1: Author the driver**

```python
"""Top-level driver for §5.4.6: launches every (algorithm, dim, instance-batch)
shard. Designed for parallel SLURM submission via `slurm/phase4_full_suite.sh`.

Usage:
  python run_phase4_full_suite.py --algorithm sage_winner --dim 10 \\
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
    args = p.parse_args()

    out_dir = Path(RESULTS_DIR) / args.algorithm / f'dim{args.dim}'
    instances = list(range(args.instance_start, args.instance_end))
    out = run_shard(args.algorithm, args.dim, instances, out_dir)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Sanity check `--help`**

```bash
python run_phase4_full_suite.py --help
```

Expected: argparse usage text listing the four winners + cma_es.

- [ ] **Step 3: Commit**

```bash
git add run_phase4_full_suite.py
git commit -m "Add Stage 4.6 driver script"
```

### Task E5: SLURM submission template

**Files:**
- Create: `slurm/phase4_full_suite.sh`

- [ ] **Step 1: Author the SLURM array template**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=p4_fs
#SBATCH --output=slurm_logs/p4_fs_%A_%a.out
#SBATCH --error=slurm_logs/p4_fs_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29       # 5 algorithms × 3 dims × 2 instance batches = 30 shards

set -euo pipefail

# 5 algorithms × 3 dims × 2 batches (0-499, 500-999) = 30 task IDs.
ALGS=(vanilla_winner neutral_winner sage_winner combined_neutral_winner cma_es)
DIMS=(5 10 20)
BATCHES=(0 500)
BATCH_SIZE=500

idx=$SLURM_ARRAY_TASK_ID
alg_i=$(( idx / 6 ))
dim_i=$(( (idx / 2) % 3 ))
batch_i=$(( idx % 2 ))

ALG=${ALGS[$alg_i]}
DIM=${DIMS[$dim_i]}
START=${BATCHES[$batch_i]}
END=$(( START + BATCH_SIZE ))

source /local/$USER/conda_envs/thesis/bin/activate
cd /home/$USER/repos/thesis

mkdir -p slurm_logs
python run_phase4_full_suite.py \
  --algorithm "$ALG" --dim "$DIM" \
  --instance-start "$START" --instance-end "$END"
```

- [ ] **Step 2: Sanity check the math**

The array index 0 should map to (alg=vanilla_winner, dim=5, start=0).
Index 29 should map to (alg=cma_es, dim=20, start=500).

```bash
for idx in 0 1 5 6 12 29; do
  python - <<PY
idx=$idx
algs=['vanilla_winner','neutral_winner','sage_winner','combined_neutral_winner','cma_es']
dims=[5,10,20]
batches=[0,500]
ai=idx//6; di=(idx//2)%3; bi=idx%2
print(f'idx={idx}: alg={algs[ai]} dim={dims[di]} start={batches[bi]}')
PY
done
```

Expected: prints six lines covering corner cases. Adjust the batch math if you want finer-grained shards (e.g., 5 batches of 200).

- [ ] **Step 3: Commit (do NOT submit yet)**

```bash
git add slurm/phase4_full_suite.sh
git commit -m "Add SLURM submission template for Stage 4.6"
```

### Task E6: Submit and monitor the SLURM job (longest pole)

**Files:**
- Run-only (writes `results_phase4_full_suite/<alg>/dim<d>/...parquet`)

- [ ] **Step 1: Sync the repo to saronite**

```bash
ssh saronite 'mkdir -p /home/$USER/repos/thesis/slurm_logs'
rsync -av --exclude='.git' --exclude='results_phase*' \
  /Users/maxharell/repos/thesis/ saronite:/home/$USER/repos/thesis/
```

- [ ] **Step 2: Submit the array**

```bash
ssh saronite 'cd /home/$USER/repos/thesis && sbatch slurm/phase4_full_suite.sh'
```

Note the job ID. Confirm `squeue -u $USER` shows 30 array tasks pending.

- [ ] **Step 3: Wait for completion and pull results back**

```bash
ssh saronite 'squeue -u $USER -h | wc -l'   # 0 when done
rsync -av saronite:/home/$USER/repos/thesis/results_phase4_full_suite/ \
  /Users/maxharell/repos/thesis/results_phase4_full_suite/
```

- [ ] **Step 4: Verify all 30 shards produced output**

```bash
find results_phase4_full_suite -name '*.parquet' | wc -l
```

Expected: 30. If fewer, inspect `slurm_logs/p4_fs_*.err` for failures.

- [ ] **Step 5: No commit — results files are git-ignored**

Confirm `.gitignore` already excludes `results_phase4_full_suite/`. If not, add it now and commit only the gitignore change.

### Task E7: Aggregation notebook

**Files:**
- Create: `analysis/phase4_full_suite.ipynb`

- [ ] **Step 1: Open Jupyter and create the notebook**

```bash
jupyter notebook analysis/phase4_full_suite.ipynb
```

- [ ] **Step 2: First cell — load and merge shards**

```python
from pathlib import Path
import pandas as pd

RESULTS = Path('/Users/maxharell/repos/thesis/results_phase4_full_suite')
shards = list(RESULTS.glob('*/dim*/*.parquet'))
print(f'found {len(shards)} shards')
df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
print(df.shape)
print(df.head())
```

Expected: shape (1000 instances × 5 seeds × 5 algorithms × 3 dims =) 75 000 rows.

- [ ] **Step 3: Second cell — per-(algorithm, dim) summary**

```python
summary = (
    df.groupby(['algorithm', 'dim']).agg(
        mean_aocc=('aocc', 'mean'),
        std_aocc=('aocc', 'std'),
        n=('aocc', 'size'),
    ).reset_index()
)
summary.to_csv('/Users/maxharell/repos/thesis/analysis/figs_phase4/p4_full_suite_summary.csv',
               index=False)
print(summary)
```

- [ ] **Step 4: Third cell — bar chart by BBOB function group**

The instance → BBOB-group mapping comes from the same logic the Stage 4 instance selector uses. Reuse `analysis/phase4_instance_selection.ipynb`'s `bbob_group_for_instance` function (copy-paste it; it's small).

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Map each MA-BBOB instance to its dominant BBOB-function group.
from analysis.phase4_instance_selection_helpers import bbob_group_for_instance
df['bbob_group'] = df['instance'].map(bbob_group_for_instance)

ALGS = ['vanilla_winner', 'neutral_winner', 'sage_winner', 'combined_neutral_winner', 'cma_es']
GROUPS = ['Sep', 'LowMod', 'HighUni', 'MMAdeq', 'MMWeak']

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, dim in zip(axes, [5, 10, 20]):
    sub = df[df.dim == dim].groupby(['algorithm', 'bbob_group']).aocc.mean().unstack()
    sub.loc[ALGS, GROUPS].plot(kind='bar', ax=ax)
    ax.set_title(f'dim = {dim}')
    ax.set_ylabel('Mean AOCC')
    ax.legend(title='BBOB group', fontsize=8)
plt.tight_layout()
plt.savefig('/Users/maxharell/repos/thesis/analysis/figs_phase4/p4_full_suite_groups.pdf',
            bbox_inches='tight')
plt.show()
```

If `analysis/phase4_instance_selection_helpers.py` does not exist, extract the small `bbob_group_for_instance` helper from `analysis/phase4_instance_selection.ipynb` into that file and add a one-line unit test under `tests/analysis/`.

- [ ] **Step 5: Fourth cell — per-dimension delta-vs-CMA-ES table**

```python
piv = summary.pivot(index='algorithm', columns='dim', values='mean_aocc').loc[ALGS]
delta = piv.subtract(piv.loc['cma_es'])
delta.to_csv('/Users/maxharell/repos/thesis/analysis/figs_phase4/p4_full_suite_delta_vs_cma.csv')
print(piv.round(3))
print('\nΔ vs CMA-ES:')
print(delta.round(3))
```

- [ ] **Step 6: Commit**

```bash
git add analysis/phase4_full_suite.ipynb analysis/figs_phase4/p4_full_suite_summary.csv analysis/figs_phase4/p4_full_suite_groups.pdf analysis/figs_phase4/p4_full_suite_delta_vs_cma.csv
git commit -m "Aggregate Stage 4.6 full-suite results into figures and tables"
```

---

## Phase F — Polish and final sweep

### Task F1: Cross-check that every spec deliverable has a saved artefact

- [ ] **Step 1: List every file the spec promises**

```bash
ls analysis/figs_phase4/p4_*.pdf analysis/figs_phase4/p4_*.csv
```

Expected (minimum): `p4_failure_rates.pdf`, `p4_failure_rate_by_gen.pdf`, `p4_failure_mode_breakdown.csv`, `p4_failure_modes.csv`, `p4_final_aocc_boxplot.pdf`, `p4_per_instance_heatmap.pdf`, `p4_convergence.pdf`, `p4_behavioural_profiles.pdf`, `p4_steering_rates.csv`, `p4_winner_metrics.csv`, `p4_winner_identity.csv`, `p4_full_suite_summary.csv`, `p4_full_suite_groups.pdf`, `p4_full_suite_delta_vs_cma.csv`.

- [ ] **Step 2: For any missing artefact, return to the relevant phase task**

Don't proceed to ch5 prose-writing until every file in the list above exists and visually checks out.

- [ ] **Step 3: Check tests still pass**

```bash
pytest tests/analysis -v
```

Expected: all passes (failure_modes, steering, code_identity).

- [ ] **Step 4: Final commit pass — anything still uncommitted**

```bash
git status
```

Expect clean.

### Task F2: Update revision_plan_2026-04-21.md

- [ ] **Step 1: Mark Stage 4 analysis items resolved**

Open `docs/thesisLatex/revision_plan_2026-04-21.md` and tick:

- `[ ] Analyse Phase 4 results.` → `[x]` with date and a one-line note pointing at `docs/superpowers/specs/2026-05-03-stage4-analysis-design.md`.
- Note 9 (quantify steering success) → `[x]` referencing `analysis/figs_phase4/p4_steering_rates.csv`.

- [ ] **Step 2: Commit**

```bash
git add docs/thesisLatex/revision_plan_2026-04-21.md
git commit -m "Mark Stage 4 analysis revision-plan items as complete"
```

---

## Self-review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| §5.4.1 Aggregate condition performance (existing artefacts + bootstrap CI) | A1, A2 |
| §5.4.2 Variance and per-instance robustness (existing artefacts) | A1 |
| §5.4.3 Failure-rate analysis (gen-binned, mode breakdown, no date-binning) | B1–B6 |
| §5.4.4 Behavioural profiles + steering-success quantification | C1–C3 |
| §5.4.5 Four best algorithms (code identity) | D1–D4 |
| §5.4.6 Full-suite generalisation experiment | E1–E7 |
| Polish (revision-plan note 34 SEM/CI; final cross-check) | A2, F1, F2 |

**Open dependencies the spec deferred:**

- §5.4.6 second baseline beyond CMA-ES — out of scope here. If decided later, add to `experiments/phase4_full_suite_config.py`'s `ALGORITHMS` dict and re-run only the new shard.
- Statistical-test choice for the headline — both bootstrap CI (A2) and MWU+Cliff's δ+Holm (existing notebook) are produced. Final selection happens at write-up time, not here.
- Directional-condition handling — explicitly excluded.

**Placeholder scan:** none — every step contains the actual code or command. The two manual steps (D4 family classification, F2 revision-plan tick-offs) are explicit about what the human does and why, with file paths.

**Type consistency:** `classify_failure(code) → (label, detail)` is used identically in B3 (definition) and B4 (cell). `steering_rate(df, *, feature, condition, vanilla, direction)` is used identically in C2 and C3. `code_metrics`, `family_prompt` signatures match between D2 and D3. Runner's `run_shard(alg_name, dim, instance_indices, out_dir)` matches between E2, E3, E4.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-stage4-analysis.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
