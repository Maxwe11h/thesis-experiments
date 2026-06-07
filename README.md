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
