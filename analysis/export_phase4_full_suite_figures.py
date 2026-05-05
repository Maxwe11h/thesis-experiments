#!/usr/bin/env python3
"""Export §5.4.6 full-suite generalisation figures.

Reads `results_phase4_full_suite/<alg>/dim<d>/*.parquet` (one parquet per
alg-dim shard, each with 5000 rows = 1000 instances x 5 seeds, plus
per-FE best-so-far curves) and produces:

  * fig_phase4_fullsuite_ecdf.pdf       (LLaMEA Fig 11 style)
  * fig_phase4_fullsuite_final_aocc.pdf (boxplot per dim)
  * analysis/figs_phase4_full_suite/p46_final_aocc_summary.csv

Styling matches `analysis/export_phase4_figures.py` (serif, pastel boxes,
dark medians).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results_phase4_full_suite"
THESIS_FIG_DIR = REPO_ROOT / "docs" / "thesisLatex" / "figures"
LOCAL_FIG_DIR = SCRIPT_DIR / "figs_phase4_full_suite"
THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Style — matches export_phase4_figures.py
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (12, 4.5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.05,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.5,
    "axes.grid": False,
})
SAVEFIG_KW = dict(bbox_inches="tight", dpi=300)

# ---------------------------------------------------------------------------
# Algorithm & display constants
# ---------------------------------------------------------------------------
ALGS = [
    "vanilla_winner",
    "neutral_winner",
    "sage_winner",
    "combined_neutral_winner",
    "cma_es",
]
ALG_LABELS = {
    "vanilla_winner": "Vanilla",
    "neutral_winner": "Neutral",
    "sage_winner": "SAGE",
    "combined_neutral_winner": "Combined",
    "cma_es": "CMA-ES",
}
ALG_COLORS = {
    "vanilla_winner": "#888888",
    "neutral_winner": "#4e79a7",
    "sage_winner": "#E63946",
    "combined_neutral_winner": "#6A4C93",
    "cma_es": "#111111",
}
ALG_LINESTYLES = {
    "vanilla_winner": "-",
    "neutral_winner": "-",
    "sage_winner": "-",
    "combined_neutral_winner": "-",
    "cma_es": "--",   # dashed to flag as external baseline
}

DIMS = [5, 10, 20]
BUDGET_FACTOR = 2000

# AOCC log-target range (matches the AOCC definition used in the runner).
TARGET_LB, TARGET_UB = 1e-8, 1e2
N_TARGETS = 51
N_FE_GRID = 100   # log-spaced FE evaluation points per dim


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_shard(alg: str, dim: int) -> pd.DataFrame:
    """Read one (alg, dim) parquet into a DataFrame keyed by instance, seed."""
    matches = list((RESULTS_DIR / alg / f"dim{dim}").glob("*.parquet"))
    if not matches:
        raise FileNotFoundError(f"missing shard for {alg} dim={dim}")
    return pd.concat([pd.read_parquet(p) for p in sorted(matches)],
                     ignore_index=True)


def stack_curves(df: pd.DataFrame) -> np.ndarray:
    """Stack the parquet `curve` column into a 2-D float32 array.

    A handful of user-coded algorithm runs (mostly combined_neutral_winner
    at d=5 / d=20 on adversarial instances) emit NaN evaluations once their
    iterates diverge. Per BBOB convention, treat those as "no progress" by
    pinning to the AOCC upper bound — this keeps both the ECDF computation
    and the recomputed AOCC well-defined.
    """
    curves = np.stack(df["curve"].to_numpy())
    return np.where(np.isnan(curves), np.float32(TARGET_UB), curves)


def _aocc(curves: np.ndarray) -> np.ndarray:
    """Per-run AOCC, identical to the runner's `_aocc` (thesis §4.5 eq 4.4)."""
    log_curve = np.log10(np.clip(curves, TARGET_LB, TARGET_UB))
    log_lb, log_ub = np.log10(TARGET_LB), np.log10(TARGET_UB)
    return (1.0 - (log_curve - log_lb) / (log_ub - log_lb)).mean(axis=1)


# ---------------------------------------------------------------------------
# ECDF / EAF computation
# ---------------------------------------------------------------------------
def compute_ecdf(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fe_grid, ecdf) where ecdf is mean fraction-of-runs that have
    hit each of N_TARGETS log-spaced targets, evaluated on N_FE_GRID
    log-spaced FE points.

    Curves are best-so-far (monotone non-increasing) so `curves[r, fe]`
    equals the minimum fitness achieved by run r within the first fe+1
    function evaluations.
    """
    budget = curves.shape[1]
    targets = np.logspace(np.log10(TARGET_UB), np.log10(TARGET_LB), N_TARGETS)
    fe_grid = np.unique(
        np.logspace(0, np.log10(budget), N_FE_GRID).astype(int)
    )
    sub = curves[:, fe_grid - 1]                                 # (n_runs, n_fe)
    hits = sub[:, :, None] <= targets[None, None, :]             # (n_runs, n_fe, n_t)
    ecdf = hits.mean(axis=(0, 2))                                # (n_fe,)
    return fe_grid, ecdf


# ---------------------------------------------------------------------------
# Figure 1 — ECDF / EAF per dim
# ---------------------------------------------------------------------------
def fig_ecdf() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, dim in zip(axes, DIMS):
        budget = BUDGET_FACTOR * dim
        for alg in ALGS:
            df = load_shard(alg, dim)
            curves = stack_curves(df)
            x, y = compute_ecdf(curves)
            ax.plot(
                x, y,
                color=ALG_COLORS[alg],
                linestyle=ALG_LINESTYLES[alg],
                linewidth=2.0,
                label=ALG_LABELS[alg],
            )
        ax.set_xlabel("function evaluations")
        ax.set_title(f"$d = {dim}$")
        ax.set_xlim(0, budget)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, which="major", color="#cccccc", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_color("#bbbbbb")

    axes[0].set_ylabel("EAF")
    axes[-1].legend(
        loc="lower right", frameon=False,
        handlelength=2.4, borderpad=0.4, labelspacing=0.3,
    )

    out = THESIS_FIG_DIR / "fig_phase4_fullsuite_ecdf.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    fig.savefig(LOCAL_FIG_DIR / "p46_ecdf.pdf", **SAVEFIG_KW)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Final-AOCC boxplot per dim
# ---------------------------------------------------------------------------
def _styled_boxplot(ax, data, labels, colors, *, widths=0.55):
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=widths,
        medianprops=dict(color="#222222", linewidth=1.6),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
        flierprops=dict(marker="o", markerfacecolor="#bab0ac",
                        markeredgecolor="none", markersize=3, alpha=0.4),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("none")
        patch.set_alpha(0.7)
    return bp


def fig_final_aocc() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    rows = []

    for ax, dim in zip(axes, DIMS):
        data, labels, colors = [], [], []
        for alg in ALGS:
            df = load_shard(alg, dim)
            curves = stack_curves(df)
            vals = _aocc(curves)
            data.append(vals)
            labels.append(ALG_LABELS[alg])
            colors.append(ALG_COLORS[alg])
            rows.append({
                "algorithm": alg,
                "label": ALG_LABELS[alg],
                "dim": dim,
                "n": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "median": float(np.median(vals)),
                "p25": float(np.quantile(vals, 0.25)),
                "p75": float(np.quantile(vals, 0.75)),
            })
        _styled_boxplot(ax, data, labels, colors)
        ax.set_title(f"$d = {dim}$")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=20)
        ax.yaxis.grid(True, which="major", color="#cccccc", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_color("#bbbbbb")

    axes[0].set_ylabel("AOCC")

    out = THESIS_FIG_DIR / "fig_phase4_fullsuite_final_aocc.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    fig.savefig(LOCAL_FIG_DIR / "p46_final_aocc_boxplot.pdf", **SAVEFIG_KW)
    plt.close(fig)
    print(f"wrote {out}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table — final AOCC summary CSV
# ---------------------------------------------------------------------------
def write_summary_table(summary: pd.DataFrame) -> None:
    out = LOCAL_FIG_DIR / "p46_final_aocc_summary.csv"
    summary.to_csv(out, index=False)
    print(f"wrote {out}")

    # Also print a wide view for eyeballing.
    pivot = summary.pivot(index="label", columns="dim", values="mean").round(4)
    pivot = pivot.reindex([ALG_LABELS[a] for a in ALGS])
    print("\nMean AOCC per (alg, dim):")
    print(pivot.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    fig_ecdf()
    summary = fig_final_aocc()
    write_summary_table(summary)


if __name__ == "__main__":
    main()
