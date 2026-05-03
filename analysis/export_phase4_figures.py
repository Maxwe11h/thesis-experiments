#!/usr/bin/env python3
"""Export Stage 4 thesis figures from results_phase4/ to PDF.

Mirrors the styling conventions in `analysis/export_figures.py`:
serif fonts, no grid, edgeless bars, pastel boxplot internals, rounded stat
callouts. Phase-4 condition colours stay as already established
(vanilla=#888888, neutral=#2E86AB, sage=#E63946, combined=#6A4C93) per the
notebook's existing scheme.

Produces:
  - fig_phase4_final_aocc.pdf
  - fig_phase4_convergence.pdf
  - fig_phase4_per_instance.pdf
  - fig_phase4_failure_rates.pdf
  - fig_phase4_failure_by_gen.pdf
  - fig_phase4_behavioural.pdf
  - fig_phase4_budget_threshold.pdf

Usage: python analysis/export_phase4_figures.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = REPO_ROOT / "docs" / "thesisLatex" / "figures"
RESULTS_DIR = REPO_ROOT / "results_phase4"

sys.path.insert(0, str(REPO_ROOT))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — matches analysis/export_figures.py
# ---------------------------------------------------------------------------
FONT_SIZE_BASE = 11
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": FONT_SIZE_BASE,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
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
STAT_BOX = dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8,
                edgecolor="#cccccc")

# ---------------------------------------------------------------------------
# Phase 4 constants
# ---------------------------------------------------------------------------
CONDITIONS = ["vanilla", "neutral", "sage", "combined_neutral"]
COND_LABELS = {
    "vanilla": "Vanilla",
    "neutral": "Neutral",
    "sage": "SAGE",
    "combined_neutral": "Combined",
}
COND_COLORS = {
    "vanilla": "#888888",
    "neutral": "#2E86AB",
    "sage": "#E63946",
    "combined_neutral": "#6A4C93",
}
SEEDS = list(range(10))
BUDGET = 500

TRAINING_INSTANCES = [22, 93, 166, 196, 203, 288, 321, 408, 480, 513,
                      528, 598, 697, 781, 784, 803, 894, 947, 951, 999]

NEUTRAL_FEATURES = [
    "intensification_ratio",
    "dimension_convergence_heterogeneity",
    "fitness_plateau_fraction",
    "avg_improvement",
    "improvement_spatial_correlation",
]
FEATURE_LABELS = {
    "intensification_ratio": "intensification\nratio",
    "dimension_convergence_heterogeneity": "dim. convergence\nheterogeneity",
    "fitness_plateau_fraction": "fitness plateau\nfraction",
    "avg_improvement": "avg.\nimprovement",
    "improvement_spatial_correlation": "improvement\nspatial corr.",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_summary() -> pd.DataFrame:
    """Concatenate summary.csv across all (condition, seed)."""
    frames = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            fp = RESULTS_DIR / cond / f"seed-{seed}" / "summary.csv"
            if not fp.exists():
                continue
            d = pd.read_csv(fp)
            d["condition"] = cond
            d["seed"] = seed
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["AOCC_valid"] = df["AOCC"].where(df["run_status"] == "success")
    df = df.sort_values(["condition", "seed", "generation"]).reset_index(drop=True)
    df["best_so_far"] = df.groupby(["condition", "seed"])["AOCC_valid"].cummax()
    return df


def load_per_instance() -> pd.DataFrame:
    """Per-(condition, seed, generation, instance) AOCC from log.jsonl `aucs`."""
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            run_dirs = list((RESULTS_DIR / cond / f"seed-{seed}").glob("run-*"))
            if not run_dirs:
                continue
            log = run_dirs[0] / "log.jsonl"
            if not log.exists():
                continue
            with open(log) as fh:
                for i, line in enumerate(fh):
                    entry = json.loads(line)
                    aucs = entry.get("metadata", {}).get("aucs")
                    if not aucs:
                        continue
                    if isinstance(aucs[0], list):
                        per_inst = [np.mean(a) for a in aucs]
                    else:
                        per_inst = aucs
                    for k, idx in enumerate(TRAINING_INSTANCES):
                        if k < len(per_inst):
                            rows.append({
                                "condition": cond, "seed": seed,
                                "generation": i, "instance": idx,
                                "aocc": per_inst[k],
                            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _styled_boxplot(ax, data, labels, colors, *, widths=0.55):
    """Produce a boxplot in the export_figures.py style: pastel face, dark
    median, grey whisker/cap, soft fliers."""
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=widths,
        medianprops=dict(color="#333333", linewidth=1.5),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
        flierprops=dict(marker="o", markerfacecolor="#bab0ac",
                        markeredgecolor="none", markersize=4, alpha=0.6),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("none")
        patch.set_alpha(0.7)
    return bp


def _strip_overlay(ax, data, colors, *, x_offsets=None, jitter=0.04, size=24):
    rng = np.random.default_rng(42)
    x_offsets = x_offsets or list(range(1, len(data) + 1))
    for x, vals, c in zip(x_offsets, data, colors):
        if len(vals) == 0:
            continue
        jx = rng.normal(0, jitter, size=len(vals))
        ax.scatter(np.full_like(vals, x, dtype=float) + jx, vals,
                   alpha=0.65, s=size, color=c, edgecolor="none", linewidth=0,
                   zorder=3)


def _bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])


# ===========================================================================
# Figure 1 — Final AOCC boxplot
# ===========================================================================
def fig_final_aocc(df: pd.DataFrame) -> None:
    final = (df.groupby(["condition", "seed"])
             .agg(final_best=("best_so_far", "last"))
             .reset_index())

    groups = [final[final.condition == c].final_best.dropna().values for c in CONDITIONS]
    colors = [COND_COLORS[c] for c in CONDITIONS]
    labels = [COND_LABELS[c] for c in CONDITIONS]

    H, p_kw = stats.kruskal(*groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    _styled_boxplot(ax, groups, labels, colors)
    _strip_overlay(ax, groups, colors)

    # Mean markers
    for i, vals in enumerate(groups):
        ax.plot([i + 1 - 0.22, i + 1 + 0.22], [vals.mean(), vals.mean()],
                color="#333333", linewidth=2.0, zorder=4)

    ax.set_ylabel("Final best-so-far AOCC")
    ax.set_xlabel("")
    ax.set_title("Final AOCC after 500 candidates (10 seeds per condition)",
                 fontweight="bold")
    ax.text(0.02, 0.04,
            f"Kruskal-Wallis  H={H:.3f},  p={p_kw:.4f}",
            transform=ax.transAxes, va="bottom",
            fontsize=FONT_SIZE_TICK, bbox=STAT_BOX)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_final_aocc.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 2 — Convergence dynamics (95% bootstrap CI ribbon)
# ===========================================================================
def fig_convergence(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for c in CONDITIONS:
        sub = df[df.condition == c]
        rows = []
        for g, gd in sub.groupby("generation"):
            vals = gd["best_so_far"].dropna().values
            if len(vals) < 2:
                continue
            lo, hi = _bootstrap_ci(vals)
            rows.append({"generation": g, "mean": vals.mean(),
                         "ci_lo": lo, "ci_hi": hi})
        curve = pd.DataFrame(rows)
        if curve.empty:
            continue
        ax.plot(curve.generation, curve["mean"],
                label=COND_LABELS[c], color=COND_COLORS[c], linewidth=1.8)
        ax.fill_between(curve.generation, curve.ci_lo, curve.ci_hi,
                        color=COND_COLORS[c], alpha=0.18, linewidth=0)

    ax.set_xlabel("Candidate generation")
    ax.set_ylabel("Best-so-far AOCC")
    ax.set_title("Convergence dynamics (mean, 95% bootstrap CI across 10 seeds)",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_convergence.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 3 — Per-instance heatmap of best-found algorithm
# ===========================================================================
def fig_per_instance(pi: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Per-instance mean AOCC of each condition's best-so-far candidate."""
    # Pick the final best-so-far candidate per (condition, seed)
    final_idx = (summary.dropna(subset=["best_so_far"])
                 .sort_values("generation")
                 .groupby(["condition", "seed"]).tail(1)
                 [["condition", "seed", "generation", "best_so_far"]])

    # Use the highest-AOCC valid candidate within each (condition, seed)
    valid = summary[summary.run_status == "success"].copy()
    best_idx = valid.loc[valid.groupby(["condition", "seed"])["AOCC"].idxmax()][
        ["condition", "seed", "generation"]]

    merged = pi.merge(best_idx, on=["condition", "seed", "generation"], how="inner")
    matrix = (merged.groupby(["condition", "instance"])["aocc"].mean()
              .unstack().loc[CONDITIONS, TRAINING_INSTANCES])

    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(matrix.values), vmax=1.0)
    ax.set_xticks(range(len(TRAINING_INSTANCES)))
    ax.set_xticklabels([f"{i}" for i in TRAINING_INSTANCES],
                       rotation=0, fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_xlabel("MA-BBOB instance index")
    ax.set_title("Per-instance mean AOCC of each condition's best-found algorithm",
                 fontweight="bold")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v < 0.85 else "#222222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Mean AOCC (10 seeds)")

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_per_instance.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 4 — Failure-rate distribution by condition
# ===========================================================================
def fig_failure_rates(df: pd.DataFrame) -> None:
    per_seed = (df.groupby(["condition", "seed"])
                .agg(fail_pct=("run_status", lambda s: 100 * (s == "failure").mean()))
                .reset_index())
    groups = [per_seed[per_seed.condition == c].fail_pct.values for c in CONDITIONS]
    colors = [COND_COLORS[c] for c in CONDITIONS]
    labels = [COND_LABELS[c] for c in CONDITIONS]

    H, p_kw = stats.kruskal(*groups)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _styled_boxplot(ax, groups, labels, colors)
    _strip_overlay(ax, groups, colors)
    ax.set_ylabel("Per-seed failure rate (%)")
    ax.set_title("Failure rate distribution by condition",
                 fontweight="bold")
    ax.text(0.02, 0.04,
            f"Kruskal-Wallis  H={H:.3f},  p={p_kw:.4f}",
            transform=ax.transAxes, va="bottom",
            fontsize=FONT_SIZE_TICK, bbox=STAT_BOX)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_failure_rates.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 5 — Failure rate by 100-candidate generation bin
# ===========================================================================
def fig_failure_by_gen(df: pd.DataFrame) -> None:
    bins = [0, 100, 200, 300, 400, 500]
    bin_labels = ["0-99", "100-199", "200-299", "300-399", "400-499"]
    df = df.copy()
    df["gen_bin"] = pd.cut(df["generation"], bins=bins, labels=bin_labels,
                           right=False, include_lowest=True)

    rate = (df.groupby(["condition", "gen_bin"], observed=True)
            .agg(n=("run_status", "size"),
                 fail=("run_status", lambda s: (s == "failure").sum()))
            .assign(rate=lambda x: 100 * x["fail"] / x["n"])
            .reset_index())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in CONDITIONS:
        sub = rate[rate.condition == c]
        ax.plot(sub.gen_bin, sub.rate, marker="o", linewidth=1.8,
                markersize=7, color=COND_COLORS[c], label=COND_LABELS[c])
    ax.set_xlabel("Candidate generation (bin of 100)")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Failure rate over time, by condition",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_failure_by_gen.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 6 — Behavioural feature distributions (5 features × 4 conditions)
# ===========================================================================
def fig_behavioural(df: pd.DataFrame) -> None:
    valid = df[df.run_status == "success"].copy()
    feats = NEUTRAL_FEATURES

    fig, axes = plt.subplots(1, len(feats), figsize=(20, 5), sharey=False)

    for ax, feat in zip(axes, feats):
        col = f"bm_{feat}"
        groups = []
        colors = []
        for c in CONDITIONS:
            vals = valid.loc[valid.condition == c, col].dropna().values
            # Light winsorisation at 1/99 percentile for visual scale
            if len(vals) > 10:
                lo, hi = np.percentile(vals, [1, 99])
                vals = vals[(vals >= lo) & (vals <= hi)]
            groups.append(vals)
            colors.append(COND_COLORS[c])

        labels = [COND_LABELS[c] for c in CONDITIONS]
        parts = ax.violinplot(groups, showmedians=False, showextrema=False,
                              widths=0.85)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_edgecolor("none")
            body.set_alpha(0.55)
        # Median line
        for i, vals in enumerate(groups):
            if len(vals) == 0:
                continue
            m = np.median(vals)
            ax.plot([i + 1 - 0.28, i + 1 + 0.28], [m, m],
                    color="#222222", linewidth=1.6, zorder=4)

        ax.set_xticks(range(1, len(CONDITIONS) + 1))
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK - 1, rotation=20,
                           ha="right")
        ax.set_title(FEATURE_LABELS[feat], fontweight="bold",
                     fontsize=FONT_SIZE_LABEL)

    fig.suptitle("Behavioural feature distributions across conditions (valid candidates only)",
                 fontweight="bold", fontsize=FONT_SIZE_TITLE, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_behavioural.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Figure 7 — Budget-to-threshold strip plot
# ===========================================================================
def fig_budget_threshold(df: pd.DataFrame) -> None:
    final = (df.groupby(["condition", "seed"])
             .agg(final_best=("best_so_far", "last"))
             .reset_index())
    threshold = final.loc[final.condition == "vanilla", "final_best"].median()

    rows = []
    for c in CONDITIONS:
        for seed in SEEDS:
            sub = df[(df.condition == c) & (df.seed == seed)]
            if sub.empty:
                continue
            mask = sub["best_so_far"] >= threshold
            if mask.any():
                gen = int(sub.loc[mask, "generation"].iloc[0])
                rows.append({"condition": c, "seed": seed, "gen": gen,
                             "reached": True})
            else:
                rows.append({"condition": c, "seed": seed, "gen": BUDGET,
                             "reached": False})
    bt = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rng = np.random.default_rng(42)
    for i, c in enumerate(CONDITIONS):
        sub = bt[bt.condition == c]
        x = i + 1 + rng.normal(0, 0.05, size=len(sub))
        # Reached seeds: filled marker
        reached = sub[sub.reached]
        not_reached = sub[~sub.reached]
        ax.scatter(x[sub.reached.values], reached.gen.values,
                   color=COND_COLORS[c], s=70, alpha=0.85,
                   edgecolor="none", zorder=3)
        ax.scatter(x[(~sub.reached).values], not_reached.gen.values,
                   facecolor="white", edgecolor=COND_COLORS[c],
                   s=70, linewidths=1.5, zorder=3)

        # Median bar (across reached seeds only)
        if len(reached):
            m = np.median(reached.gen.values)
            ax.plot([i + 1 - 0.24, i + 1 + 0.24], [m, m],
                    color="#333333", linewidth=2.0, zorder=4)
        # Reached fraction annotation
        n_r = len(reached)
        n_t = len(sub)
        ax.text(i + 1, BUDGET + 18, f"{n_r}/{n_t} reached",
                ha="center", fontsize=FONT_SIZE_TICK)

    ax.set_xticks(range(1, len(CONDITIONS) + 1))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Candidate generation to reach threshold")
    ax.set_xlabel("")
    ax.set_ylim(-15, BUDGET + 60)
    ax.set_title(f"Budget to reach vanilla median final AOCC ({threshold:.3f})",
                 fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor="#666666", edgecolor="none",
                       label="Threshold reached"),
        mpatches.Patch(facecolor="white", edgecolor="#666666",
                       label="Never reached (capped at 500)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_phase4_budget_threshold.pdf"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("=" * 60)
    print("Exporting Stage 4 thesis figures to PDF")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"  WARNING: {RESULTS_DIR} not found")
        return

    print("\n[Stage 4] Loading summary.csv across conditions and seeds...")
    df = load_summary()
    print(f"  loaded {len(df)} candidate rows "
          f"({df.condition.nunique()} conditions, {df.seed.nunique()} seeds)")

    print("\n[Stage 4] fig_phase4_final_aocc.pdf")
    fig_final_aocc(df)

    print("[Stage 4] fig_phase4_convergence.pdf")
    fig_convergence(df)

    print("[Stage 4] fig_phase4_failure_rates.pdf")
    fig_failure_rates(df)

    print("[Stage 4] fig_phase4_failure_by_gen.pdf")
    fig_failure_by_gen(df)

    print("[Stage 4] fig_phase4_behavioural.pdf")
    fig_behavioural(df)

    print("[Stage 4] fig_phase4_budget_threshold.pdf")
    fig_budget_threshold(df)

    print("\n[Stage 4] Loading per-instance log.jsonl ...")
    pi = load_per_instance()
    print(f"  loaded {len(pi)} per-instance AOCC rows")
    print("[Stage 4] fig_phase4_per_instance.pdf")
    fig_per_instance(pi, df)

    produced = sorted(FIGURES_DIR.glob("fig_phase4_*.pdf"))
    print("\n" + "=" * 60)
    print(f"Done. Produced {len(produced)} Stage 4 figures.")
    for p in produced:
        print(f"  {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
