#!/usr/bin/env python3
"""
Export thesis figures from notebook analysis code to PDF files.

Produces:
  - fig_model_ranking.pdf         : bar chart of best AOCC by model (phase1_model_ranking)
  - fig_model_convergence.pdf     : best-so-far convergence curves by model (phase1_model_ranking)
  - fig_failure_modes.pdf         : failure mode breakdown by model (phase1_model_ranking)
  - fig_spearman_heatmap.pdf      : Spearman correlation heatmap (phase1_behavior_analysis)
  - fig_format_boxplot.pdf        : boxplot of best AOCC by feedback format (phase3_feedback_analysis)
  - fig_condition_ranking.pdf     : horizontal bar chart of all 29 conditions (phase3_feedback_analysis)
  - fig_convergence_by_format.pdf : convergence curves by format per feature (phase3_feedback_analysis)
  - fig_steering_shifts.pdf       : behavioral shift bar chart by format (phase3_feedback_analysis)
  - fig_mabbob_instances.pdf      : MA-BBOB instance selection visualization (mabbob_instance_selection)

Usage:
    python export_figures.py

Data paths (relative to this script's directory):
    ../results_phase1/          — Phase 1 experiment results (10 models x 5 seeds)
    ../results_phase3/          — Phase 3 experiment results (29 conditions x 5 seeds)
    ../BLADE/iohblade/benchmarks/BBOB/mabbob/weights.csv  — MA-BBOB instance weights
"""

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = REPO_ROOT / "docs" / "thesisProposalLatex" / "figures"
RESULTS_PHASE1 = REPO_ROOT / "results_phase1"
RESULTS_PHASE3 = REPO_ROOT / "results_phase3"
WEIGHTS_CSV = REPO_ROOT / "BLADE" / "iohblade" / "benchmarks" / "BBOB" / "mabbob" / "weights.csv"

# Add repo root to path so we can import experiments.feedback
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style: soft pastel palette (Tableau-inspired)
# ---------------------------------------------------------------------------
PASTEL_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

# Consistent font sizes
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

# ===================================================================
# Phase 1 constants and helpers
# ===================================================================
MODELS = {
    "qwen3.5-4b":           {"family": "Qwen",     "size": "3-4B"},
    "qwen3.5-9b":           {"family": "Qwen",     "size": "7-9B"},
    "qwen3.5-27b":          {"family": "Qwen",     "size": "24-27B"},
    "rnj-1-8b":             {"family": "RnJ",      "size": "7-9B"},
    "devstral-small-2-24b": {"family": "Devstral", "size": "24-27B"},
    "olmo3-7b":             {"family": "OLMo",     "size": "7-9B"},
    "olmo3-32b":            {"family": "OLMo",     "size": "30-32B"},
    "granite4-3b":          {"family": "Granite",  "size": "3-4B"},
    "gemini-3-pro":         {"family": "Gemini",   "size": "API"},
    "gemini-3-flash":       {"family": "Gemini",   "size": "API"},
}

N_SEEDS = 5
BUDGET = 100
N_INSTANCES = 10
EVAL_SEEDS = 5

# Softer pastel model colors
MODEL_COLORS = {
    "qwen3.5-4b":           "#4e79a7",
    "qwen3.5-9b":           "#76a4c5",
    "qwen3.5-27b":          "#a4c8e1",
    "olmo3-7b":             "#59a14f",
    "olmo3-32b":            "#8cc87f",
    "rnj-1-8b":             "#b07aa1",
    "devstral-small-2-24b": "#9c755f",
    "granite4-3b":          "#bab0ac",
    "gemini-3-pro":         "#f28e2b",
    "gemini-3-flash":       "#ffbe7d",
}

# Phase 3 constants
FEATURES = [
    "avg_improvement",
    "intensification_ratio",
    "fitness_plateau_fraction",
    "step_size_autocorrelation",
    "improvement_spatial_correlation",
    "half_convergence_time",
    "fitness_autocorrelation",
    "x_spread_early",
    "longest_no_improvement_streak",
    "dimension_convergence_heterogeneity",
]
FORMATS = ["neutral", "directional", "comparative"]
COMPARATIVE_EXCLUDE = {"longest_no_improvement_streak"}
FORMAT_COLORS = {
    "neutral": "#4e79a7",
    "directional": "#f28e2b",
    "comparative": "#59a14f",
}
FEATURE_SHORT = {
    "avg_improvement": "avg_impr",
    "intensification_ratio": "intens_ratio",
    "fitness_plateau_fraction": "plateau_frac",
    "step_size_autocorrelation": "step_autocorr",
    "improvement_spatial_correlation": "impr_spatial",
    "half_convergence_time": "half_conv",
    "fitness_autocorrelation": "fit_autocorr",
    "x_spread_early": "x_spread",
    "longest_no_improvement_streak": "no_impr_streak",
    "dimension_convergence_heterogeneity": "dim_conv_het",
}

# Vanilla baseline AOCC for Phase 3 reference line
VANILLA_AOCC = 0.862


def formats_for_feature(feat):
    if feat in COMPARATIVE_EXCLUDE:
        return ["neutral", "directional"]
    return FORMATS


def parse_fitness(val):
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return float("-inf")


def parse_fitness_nan(val):
    """Like parse_fitness but returns NaN for failures (used by phase3)."""
    try:
        f = float(val)
        return f if not (math.isinf(f) and f < 0) else np.nan
    except (TypeError, ValueError):
        return np.nan


def categorize_failure(entry):
    code = entry.get("code", "")
    feedback = entry.get("feedback", "")
    if not code or not code.strip():
        return "no_code"
    if "class " not in code:
        return "no_class"
    if "unexpected keyword argument 'budget'" in feedback:
        return "init_missing_budget"
    if "unexpected keyword argument 'dim'" in feedback:
        return "init_missing_dim"
    if "positional argument" in feedback and "bounds" in feedback:
        return "call_expects_bounds"
    if "__init__" in feedback and "argument" in feedback:
        return "init_signature_other"
    if "__call__" in feedback and "argument" in feedback:
        return "call_signature_other"
    if "import" in feedback.lower() or "ModuleNotFoundError" in feedback:
        return "import_error"
    if "SyntaxError" in feedback or "IndentationError" in feedback:
        return "syntax_error"
    return "runtime_error"


# ===================================================================
# Data loaders
# ===================================================================

def load_phase1_run(model_tag, seed):
    seed_dir = RESULTS_PHASE1 / model_tag / f"seed-{seed}"
    run_dirs = sorted(seed_dir.glob("run-*"))
    if not run_dirs:
        return []
    log_file = run_dirs[0] / "log.jsonl"
    if not log_file.exists():
        return []
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_phase1():
    """Load Phase 1 results into a DataFrame."""
    rows = []
    for model_tag in MODELS:
        for seed in range(N_SEEDS):
            entries = load_phase1_run(model_tag, seed)
            for i, entry in enumerate(entries):
                fitness = parse_fitness(entry.get("fitness"))
                failed = np.isinf(fitness) and fitness < 0
                meta = entry.get("metadata", {})
                bf = meta.get("behavioral_features", {})
                row = {
                    "model": model_tag,
                    "seed": seed,
                    "evaluation": i,
                    "fitness": fitness,
                    "failed": failed,
                    "name": entry.get("name", ""),
                    "code": entry.get("code", ""),
                    "feedback": entry.get("feedback", ""),
                }
                for k, v in bf.items():
                    col_name = "fitness_autocorrelation" if k == "fitness_autocorrelation_lag1" else k
                    row[f"bm_{col_name}"] = v
                rows.append(row)
    return pd.DataFrame(rows)


def load_phase3():
    """Load Phase 3 results into a DataFrame."""
    conditions = []
    for feat in FEATURES:
        for fmt in FORMATS:
            if fmt == "comparative" and feat in COMPARATIVE_EXCLUDE:
                continue
            conditions.append(f"{fmt}-{feat}")

    rows = []
    for cond in conditions:
        parts = cond.split("-", 1)
        fmt, feat = parts[0], parts[1]
        for seed in range(N_SEEDS):
            seed_dir = RESULTS_PHASE3 / cond / f"seed-{seed}"
            run_dirs = sorted(seed_dir.glob("run-*"))
            if not run_dirs:
                continue
            log_file = run_dirs[0] / "log.jsonl"
            if not log_file.exists():
                continue
            with open(log_file) as fh:
                for i, line in enumerate(fh):
                    entry = json.loads(line.strip())
                    fitness = parse_fitness_nan(entry.get("fitness"))
                    meta = entry.get("metadata", {})
                    aucs = meta.get("aucs", [])
                    bf = meta.get("behavioral_features", {})
                    row = {
                        "condition": cond,
                        "format": fmt,
                        "feature": feat,
                        "seed": seed,
                        "evaluation": i,
                        "fitness": fitness,
                        "failed": np.isnan(fitness),
                        "name": entry.get("name", ""),
                        "aucs": aucs,
                    }
                    for bk, bv in bf.items():
                        row[f"bf_{bk}"] = bv
                    rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# Figure 1: Model Screening (ranking + convergence, combined)
# ===================================================================

def fig_model_screening(df):
    """Combined figure: (a) bar chart of best AOCC, (b) convergence curves."""
    # --- Compute ranking data ---
    best_per_seed = df.groupby(["model", "seed"])["fitness"].max().reset_index()
    best_per_seed.columns = ["model", "seed", "best_aocc"]
    best_per_seed["best_aocc"] = best_per_seed["best_aocc"].replace(-np.inf, np.nan)

    summary = best_per_seed.groupby("model")["best_aocc"].agg(["mean", "std"]).reset_index()
    summary.columns = ["model", "aocc_mean", "aocc_std"]
    summary = summary.sort_values("aocc_mean", ascending=False).reset_index(drop=True)

    # --- Compute convergence data ---
    def compute_best_so_far(model_tag):
        curves = []
        for seed in range(N_SEEDS):
            sub = df[(df["model"] == model_tag) & (df["seed"] == seed)].sort_values("evaluation")
            fitness_vals = sub["fitness"].values.copy().astype(float)
            fitness_vals[np.isinf(fitness_vals) & (fitness_vals < 0)] = np.nan
            best_so_far = pd.Series(fitness_vals).expanding().max().values
            curves.append(best_so_far)
        max_len = max(len(c) for c in curves) if curves else 0
        padded = []
        for c in curves:
            if len(c) < max_len:
                c = np.concatenate([c, np.full(max_len - len(c), c[-1] if len(c) > 0 else np.nan)])
            padded.append(c)
        return np.array(padded)

    # --- Combined figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel (a): Model ranking bar chart
    models_sorted = summary["model"].tolist()
    y_pos = np.arange(len(models_sorted))
    colors = [MODEL_COLORS[m] for m in models_sorted]
    ax1.barh(y_pos, summary["aocc_mean"], xerr=summary["aocc_std"],
             color=colors, edgecolor="none", capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_sorted)
    ax1.set_xlabel("Mean Best AOCC")
    ax1.set_title("(a) Model Ranking (mean ± std across 5 seeds)",
                  fontweight="bold")
    ax1.invert_yaxis()

    # Panel (b): Convergence curves
    for model_tag in MODELS:
        curves = compute_best_so_far(model_tag)
        if curves.size == 0:
            continue
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        x = np.arange(1, len(mean_curve) + 1)
        color = MODEL_COLORS[model_tag]
        ax2.plot(x, mean_curve, label=model_tag, color=color, linewidth=1.5)
        ax2.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.15, color=color)
    ax2.set_xlabel("Evaluation")
    ax2.set_ylabel("Best-so-far AOCC")
    ax2.set_title("(b) Convergence (mean ± 1 std across 5 seeds)",
                  fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=FONT_SIZE_LEGEND,
               borderaxespad=0, frameon=True)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_model_screening.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")
    return summary


# ===================================================================
# Figure 2: t-SNE of Behavioral Space (Phase 1)
# ===================================================================

def fig_tsne_behavioral(df):
    """Two-panel t-SNE: (a) colored by model, (b) colored by AOCC."""
    df_valid = df[~df["failed"]].copy()
    bm_cols = [c for c in df_valid.columns if c.startswith("bm_")]
    for col in bm_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    # Drop metrics with >50% NaN, then drop rows with any remaining NaN
    nan_frac = df_valid[bm_cols].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    proj_data = df_valid[bm_cols + ["fitness", "model"]].dropna(subset=bm_cols)

    X = proj_data[bm_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = X_scaled.shape[0]
    perp = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_scaled)

    model_list = proj_data["model"].values
    fitness_vals = proj_data["fitness"].values

    # Sort models by mean AOCC for legend ordering
    model_means = proj_data.groupby("model")["fitness"].mean().sort_values(ascending=False)
    models_sorted = model_means.index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Panel (a): colored by model
    for model in models_sorted:
        mask = model_list == model
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=MODEL_COLORS[model], label=model,
                    s=30, alpha=0.6, edgecolors="w", linewidths=0.2)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.set_title("(a) Colored by model", fontweight="bold")
    ax1.legend(fontsize=FONT_SIZE_LEGEND, loc="best", ncol=2,
               markerscale=1.5, framealpha=0.9)

    # Panel (b): colored by AOCC
    order = np.argsort(fitness_vals)  # plot low AOCC first so high on top
    sc = ax2.scatter(X_tsne[order, 0], X_tsne[order, 1],
                     c=fitness_vals[order], cmap="viridis",
                     s=30, alpha=0.6, edgecolors="w", linewidths=0.2,
                     vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax2, label="AOCC", shrink=0.8)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("(b) Colored by AOCC", fontweight="bold")

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_tsne_behavioral.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figures 2b-d: Feature ranking (3 separate figures, each self-sorted)
# ===================================================================

_FEAT_CATEGORIES = {
    "Exploration & Diversity": [
        "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct"],
    "Exploitation": [
        "avg_distance_to_best", "intensification_ratio", "avg_exploitation_pct"],
    "Convergence": [
        "average_convergence_rate", "avg_improvement", "success_rate"],
    "Stagnation": [
        "longest_no_improvement_streak", "last_improvement_fraction"],
    "Step-Size & Movement": [
        "step_size_mean", "step_size_std", "step_size_trend", "directional_persistence"],
    "Information-Theoretic": [
        "fitness_sample_entropy", "fitness_permutation_entropy",
        "fitness_autocorrelation", "fitness_lempel_ziv_complexity"],
    "Early/Late Dynamics": [
        "x_spread_early", "x_spread_late", "spread_ratio", "centroid_drift",
        "f_range_early", "f_range_late", "f_range_ratio"],
    "Novel Features": [
        "improvement_spatial_correlation", "improvement_burstiness",
        "dimension_convergence_heterogeneity", "step_size_autocorrelation",
        "fitness_plateau_fraction", "half_convergence_time"],
}
_CAT_COLORS = {
    "Exploration & Diversity": "#4e79a7", "Exploitation": "#f28e2b",
    "Convergence": "#59a14f", "Stagnation": "#e15759",
    "Step-Size & Movement": "#b07aa1", "Information-Theoretic": "#9c755f",
    "Early/Late Dynamics": "#ff9da7", "Novel Features": "#76b7b2",
}
_FEAT_TO_CAT = {}
for _cat, _feats in _FEAT_CATEGORIES.items():
    for _f in _feats:
        _FEAT_TO_CAT[f"bm_{_f}"] = _cat

# Shared figsize so all three render at the same width in LaTeX
_FIG_W, _FIG_H = 14, 10


def _cat_color(feat):
    return _CAT_COLORS.get(_FEAT_TO_CAT.get(feat, ""), "#bab0ac")


def _cat_legend(ax):
    elems = [mpatches.Patch(facecolor=c, edgecolor="none", label=cat)
             for cat, c in _CAT_COLORS.items()]
    ax.legend(handles=elems, loc="lower right", fontsize=FONT_SIZE_LEGEND)


def fig_spearman(df):
    """Spearman rho with AOCC, sorted by |rho|."""
    df_valid = df[~df["failed"]].copy()
    bm_cols = [c for c in df_valid.columns if c.startswith("bm_")]
    for col in bm_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    nan_frac = df_valid[bm_cols].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    corr_data = df_valid[bm_cols + ["fitness"]].dropna()

    rows = []
    for col in bm_cols:
        valid = corr_data[[col, "fitness"]].dropna()
        if len(valid) > 3:
            r, p = stats.spearmanr(valid[col], valid["fitness"])
            rows.append({"feature": col, "rho": r, "abs_rho": abs(r),
                         "p_bonf": min(p * len(bm_cols), 1.0)})
    rho_df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    rho_df["sig"] = rho_df["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")))

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = [_cat_color(f) for f in rho_df["feature"]]
    ax.barh(range(len(rho_df)), rho_df["rho"].values, color=colors, edgecolor="none")
    for i, (_, row) in enumerate(rho_df.iterrows()):
        offset = 0.01 if row["rho"] >= 0 else -0.01
        ha = "left" if row["rho"] >= 0 else "right"
        ax.text(row["rho"] + offset, i, row["sig"], va="center", ha=ha,
                fontsize=FONT_SIZE_LEGEND)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(rho_df)))
    ax.set_yticklabels([f.replace("bm_", "") for f in rho_df["feature"]],
                       fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Spearman $\\rho$ with AOCC")
    ax.set_title("Spearman Correlation with AOCC (Bonferroni-corrected)",
                 fontweight="bold")
    ax.invert_yaxis()
    _cat_legend(ax)
    outpath = FIGURES_DIR / "fig_spearman.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def fig_rf_importance(df):
    """RF permutation importance, sorted by importance."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    df_valid = df[~df["failed"]].copy()
    bm_cols = [c for c in df_valid.columns if c.startswith("bm_")]
    for col in bm_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    nan_frac = df_valid[bm_cols].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    rf_data = df_valid[bm_cols + ["fitness"]].dropna()
    X = rf_data[bm_cols].values
    y = rf_data["fitness"].values

    rf = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": bm_cols, "importance": perm_imp.importances_mean,
        "std": perm_imp.importances_std,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = [_cat_color(f) for f in perm_df["feature"]]
    ax.barh(range(len(perm_df)), perm_df["importance"].values,
            xerr=perm_df["std"].values, color=colors, edgecolor="none", capsize=3)
    ax.set_yticks(range(len(perm_df)))
    ax.set_yticklabels([f.replace("bm_", "") for f in perm_df["feature"]],
                       fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("RF Permutation Importance (OOB $R^2$ = {:.3f})".format(rf.oob_score_),
                 fontweight="bold")
    ax.invert_yaxis()
    _cat_legend(ax)
    outpath = FIGURES_DIR / "fig_rf_importance.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def fig_ks_effect(df):
    """KS statistic (top 25% vs bottom 25%), sorted by KS, winsorized."""
    df_valid = df[~df["failed"]].copy()
    bm_cols = [c for c in df_valid.columns if c.startswith("bm_")]
    for col in bm_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    nan_frac = df_valid[bm_cols].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()

    # Winsorize at 1st/99th percentile (matches notebook methodology)
    df_ks = df_valid.copy()
    for col in bm_cols:
        vals = df_ks[col].dropna()
        if len(vals) > 10:
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            df_ks[col] = df_ks[col].clip(lower=lo, upper=hi)

    q75 = df_valid["fitness"].quantile(0.75)
    q25 = df_valid["fitness"].quantile(0.25)
    top25 = df_ks[df_ks["fitness"] >= q75]
    bot25 = df_ks[df_ks["fitness"] <= q25]

    ks_list = []
    for col in bm_cols:
        t = top25[col].dropna()
        b = bot25[col].dropna()
        if len(t) > 3 and len(b) > 3:
            ks_stat, _ = stats.ks_2samp(t, b)
            ks_list.append({"feature": col, "ks_stat": ks_stat})
    ks_df = pd.DataFrame(ks_list).sort_values("ks_stat", ascending=False)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = [_cat_color(f) for f in ks_df["feature"]]
    ax.barh(range(len(ks_df)), ks_df["ks_stat"].values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(ks_df)))
    ax.set_yticklabels([f.replace("bm_", "") for f in ks_df["feature"]],
                       fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("KS Statistic")
    ax.set_title("KS Distributional Effect (top 25% vs bottom 25%, winsorized)",
                 fontweight="bold")
    ax.invert_yaxis()
    _cat_legend(ax)
    outpath = FIGURES_DIR / "fig_ks_effect.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 3: Failure Modes (Phase 1)
# ===================================================================

def fig_failure_modes(df, summary):
    """Stacked bar chart of failure categories per model."""
    failed_df = df[df["failed"]].copy()
    failed_df["failure_type"] = failed_df.apply(categorize_failure, axis=1)

    failure_counts = failed_df.groupby(["model", "failure_type"]).size().unstack(fill_value=0)
    failure_counts = failure_counts.reindex(summary["model"].tolist(), fill_value=0)

    # Use pastel palette for the stacked segments
    n_cats = len(failure_counts.columns)
    cat_colors = (PASTEL_PALETTE * ((n_cats // len(PASTEL_PALETTE)) + 1))[:n_cats]

    fig, ax = plt.subplots(figsize=(14, 6))
    failure_counts.plot(kind="bar", stacked=True, ax=ax,
                        color=cat_colors, edgecolor="none", linewidth=0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Failures")
    ax.set_title("Failure Categories per Model (all seeds pooled)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)
    plt.xticks(rotation=45, ha="right")

    outpath = FIGURES_DIR / "fig_failure_modes.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 4: Spearman Correlation Heatmap (Phase 1 behavior)
# ===================================================================

def fig_spearman_heatmap(df):
    """Bar chart of Spearman rho of each behavioral metric with AOCC."""
    # Keep only valid (non-failed) candidates
    df_valid = df[~df["failed"]].copy()
    bm_cols = [c for c in df_valid.columns if c.startswith("bm_")]
    for col in bm_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    df_valid = df_valid.dropna(subset=bm_cols, how="all").reset_index(drop=True)

    # Drop metrics with >50% NaN
    nan_frac = df_valid[bm_cols].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()

    # Feature category definitions
    FEATURE_CATEGORIES = {
        "Exploration & Diversity": [
            "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
        ],
        "Exploitation": [
            "avg_distance_to_best", "intensification_ratio", "avg_exploitation_pct",
        ],
        "Convergence": [
            "average_convergence_rate", "avg_improvement", "success_rate",
        ],
        "Stagnation": [
            "longest_no_improvement_streak", "last_improvement_fraction",
        ],
        "Step-Size & Movement": [
            "step_size_mean", "step_size_std", "step_size_trend", "directional_persistence",
        ],
        "Information-Theoretic": [
            "fitness_sample_entropy", "fitness_permutation_entropy",
            "fitness_autocorrelation", "fitness_lempel_ziv_complexity",
        ],
        "Early/Late Dynamics": [
            "x_spread_early", "x_spread_late", "spread_ratio", "centroid_drift",
            "f_range_early", "f_range_late", "f_range_ratio",
        ],
        "Novel Features": [
            "improvement_spatial_correlation", "improvement_burstiness",
            "dimension_convergence_heterogeneity", "step_size_autocorrelation",
            "fitness_plateau_fraction", "half_convergence_time",
        ],
    }
    CATEGORY_COLORS = {
        "Exploration & Diversity": "#4e79a7",
        "Exploitation": "#f28e2b",
        "Convergence": "#59a14f",
        "Stagnation": "#e15759",
        "Step-Size & Movement": "#b07aa1",
        "Information-Theoretic": "#9c755f",
        "Early/Late Dynamics": "#ff9da7",
        "Novel Features": "#76b7b2",
    }
    feat_to_cat = {}
    for cat, feats in FEATURE_CATEGORIES.items():
        for f in feats:
            feat_to_cat[f"bm_{f}"] = cat

    # Compute Spearman correlations with fitness
    corr_data = df_valid[bm_cols + ["fitness"]].dropna()
    rho_list = []
    for col in bm_cols:
        valid = corr_data[[col, "fitness"]].dropna()
        if len(valid) > 3:
            r, p = stats.spearmanr(valid[col], valid["fitness"])
            n_tests = len(bm_cols)
            rho_list.append({
                "feature": col, "rho": r, "p": p,
                "p_bonf": min(p * n_tests, 1.0),
                "category": feat_to_cat.get(col, "Unknown"),
            })

    rho_df = pd.DataFrame(rho_list).sort_values("rho", key=abs, ascending=False)
    rho_df["sig"] = rho_df["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = [CATEGORY_COLORS.get(row["category"], "#bab0ac") for _, row in rho_df.iterrows()]
    ax.barh(range(len(rho_df)), rho_df["rho"].values, color=colors, edgecolor="none")
    labels_bar = [r.replace("bm_", "") for r in rho_df["feature"]]
    ax.set_yticks(range(len(rho_df)))
    ax.set_yticklabels(labels_bar, fontsize=FONT_SIZE_TICK)
    for i, (_, row) in enumerate(rho_df.iterrows()):
        offset = 0.01 if row["rho"] >= 0 else -0.01
        ha = "left" if row["rho"] >= 0 else "right"
        ax.text(row["rho"] + offset, i, row["sig"], va="center", ha=ha,
                fontsize=FONT_SIZE_LEGEND)
    ax.set_xlabel("Spearman rho with AOCC")
    ax.set_title("Behavioral Feature Correlation with AOCC (Bonferroni-corrected)",
                 fontweight="bold")
    ax.axvline(0, color="k", lw=0.5)
    legend_elements = [mpatches.Patch(facecolor=c, edgecolor="none", label=cat)
                       for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=FONT_SIZE_LEGEND)
    ax.invert_yaxis()

    outpath = FIGURES_DIR / "fig_spearman_heatmap.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 5: Format Boxplot (Phase 3)
# ===================================================================

def fig_format_boxplot(df3):
    """Boxplot of best AOCC per seed, grouped by feedback format."""
    best = df3.groupby(["condition", "format", "feature", "seed"])["fitness"].max().reset_index()
    best.columns = ["condition", "format", "feature", "seed", "best_aocc"]

    # Only use conditions where all 3 formats exist (exclude longest_no_improvement_streak)
    best_3fmt = best[~best["feature"].isin(COMPARATIVE_EXCLUDE)]
    format_data = [best_3fmt[best_3fmt["format"] == fmt]["best_aocc"].dropna().values
                   for fmt in FORMATS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot with clean styling
    bp = ax1.boxplot(format_data, labels=FORMATS, patch_artist=True, widths=0.5,
                     medianprops=dict(color="#333333", linewidth=1.5),
                     whiskerprops=dict(color="#666666"),
                     capprops=dict(color="#666666"),
                     flierprops=dict(marker="o", markerfacecolor="#bab0ac",
                                     markeredgecolor="none", markersize=4, alpha=0.6))
    for patch, fmt in zip(bp["boxes"], FORMATS):
        patch.set_facecolor(FORMAT_COLORS[fmt])
        patch.set_edgecolor("none")
        patch.set_alpha(0.7)
    ax1.set_ylabel("Best AOCC")
    ax1.set_title("Distribution of Best AOCC by Format\n(features with all 3 formats)",
                  fontweight="bold")

    # Strip plot overlay
    for i, fmt in enumerate(FORMATS):
        vals = best_3fmt[best_3fmt["format"] == fmt]["best_aocc"].dropna().values
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
        ax1.scatter(np.full_like(vals, i + 1) + jitter, vals,
                    alpha=0.5, s=20, color=FORMAT_COLORS[fmt],
                    edgecolor="none", linewidth=0)

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*format_data)
    ax1.text(0.02, 0.98, f"Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.4f}",
             transform=ax1.transAxes, va="top", fontsize=FONT_SIZE_TICK,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8,
                       edgecolor="#cccccc"))

    # Ranking: for each (feature, seed), rank the formats
    rank_rows = []
    for feat in FEATURES:
        fmts = formats_for_feature(feat)
        for seed in range(N_SEEDS):
            seed_vals = {}
            for fmt in fmts:
                sub = best[(best["feature"] == feat) & (best["format"] == fmt) &
                           (best["seed"] == seed)]
                if not sub.empty:
                    seed_vals[fmt] = sub["best_aocc"].values[0]
            if len(seed_vals) == len(fmts):
                ranked = sorted(seed_vals.items(), key=lambda x: -x[1])
                for rank, (fmt, val) in enumerate(ranked):
                    rank_rows.append({"feature": feat, "seed": seed,
                                      "format": fmt, "rank": rank + 1})
    rank_df = pd.DataFrame(rank_rows)
    mean_rank = rank_df.groupby("format")["rank"].mean()
    win_counts = rank_df[rank_df["rank"] == 1].groupby("format").size()

    ax2.bar(FORMATS, [mean_rank.get(f, 0) for f in FORMATS],
            color=[FORMAT_COLORS[f] for f in FORMATS], edgecolor="none")
    ax2.set_ylabel("Mean Rank (1=best)")
    ax2.set_title("Mean Rank Across Feature x Seed Combinations",
                  fontweight="bold")
    for i, fmt in enumerate(FORMATS):
        wins = win_counts.get(fmt, 0)
        total = len(rank_df[rank_df["format"] == fmt])
        ax2.text(i, mean_rank.get(fmt, 0) + 0.02, f"{wins}/{total} wins",
                 ha="center", fontsize=FONT_SIZE_TICK)

    outpath = FIGURES_DIR / "fig_format_boxplot.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")
    return best


# ===================================================================
# Figure 6: Condition Ranking (Phase 3)
# ===================================================================

def fig_condition_ranking(df3, best):
    """Horizontal bar chart of all 29 conditions ranked by mean best AOCC."""
    fail_per_seed = df3.groupby(["condition", "format", "feature", "seed"])["failed"].mean().reset_index()
    fail_per_seed.columns = ["condition", "format", "feature", "seed", "failure_rate"]

    cond_summary = best.groupby(["condition", "format", "feature"])["best_aocc"].agg(
        ["mean", "std"]).reset_index()
    cond_summary.columns = ["condition", "format", "feature", "aocc_mean", "aocc_std"]

    fail_agg = fail_per_seed.groupby("condition")["failure_rate"].mean().reset_index()
    fail_agg.columns = ["condition", "fail_rate"]
    cond_summary = cond_summary.merge(fail_agg, on="condition")
    cond_summary = cond_summary.sort_values("aocc_mean", ascending=False).reset_index(drop=True)

    # Insert vanilla baseline as a grey bar at the top
    n_conds = len(cond_summary)
    fig, ax = plt.subplots(figsize=(12, 15))
    # Add vanilla as position 0
    y_pos = np.arange(n_conds + 1)
    all_labels = ["vanilla (baseline)"] + cond_summary["condition"].tolist()
    all_means = [VANILLA_AOCC] + cond_summary["aocc_mean"].tolist()
    all_stds = [0.097] + cond_summary["aocc_std"].tolist()
    all_colors = ["#cccccc"] + [FORMAT_COLORS[row["format"]] for _, row in cond_summary.iterrows()]

    ax.barh(y_pos, all_means, xerr=all_stds,
            color=all_colors, edgecolor="none", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=FONT_SIZE_LEGEND)
    ax.set_xlabel("Mean Best AOCC")
    ax.set_title("All 29 Conditions Ranked by Mean Best AOCC",
                 fontweight="bold")
    ax.invert_yaxis()

    # Vertical dashed line at vanilla baseline (no text label)
    ax.axvline(x=VANILLA_AOCC, color="#888888", linestyle="--", linewidth=1.0, zorder=0)

    legend_elements = [mpatches.Patch(facecolor=FORMAT_COLORS[f], edgecolor="none", label=f)
                       for f in FORMATS]
    legend_elements.append(mpatches.Patch(facecolor="#cccccc", edgecolor="none", label="vanilla"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=FONT_SIZE_LEGEND)

    outpath = FIGURES_DIR / "fig_condition_ranking.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 7: Convergence by Format (Phase 3)
# ===================================================================

def fig_convergence_by_format(df3):
    """Per-feature convergence curves, one curve per format (2x5 grid)."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharey=True)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURES):
        ax = axes_flat[idx]
        fmts = formats_for_feature(feat)
        for fmt in fmts:
            curves = []
            for seed in range(N_SEEDS):
                sub = df3[(df3["feature"] == feat) & (df3["format"] == fmt) &
                          (df3["seed"] == seed)]
                sub = sub.sort_values("evaluation")
                if sub.empty:
                    continue
                vals = sub["fitness"].values.copy()
                bsf = pd.Series(vals).expanding().max().values
                curves.append(bsf)

            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            padded = [np.concatenate([c, np.full(max_len - len(c), c[-1])]) for c in curves]
            arr = np.array(padded)
            mean_c = np.nanmean(arr, axis=0)
            std_c = np.nanstd(arr, axis=0)
            x = np.arange(1, len(mean_c) + 1)
            ax.plot(x, mean_c, label=fmt, color=FORMAT_COLORS[fmt], linewidth=1.5)
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            alpha=0.15, color=FORMAT_COLORS[fmt])

        ax.set_title(FEATURE_SHORT[feat], fontsize=FONT_SIZE_TICK, fontweight="bold")
        ax.set_xlabel("Evaluation")
        if idx % 5 == 0:
            ax.set_ylabel("Best-so-far AOCC")
        ax.set_ylim(0.4, 1.0)

    axes_flat[-1].legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)

    outpath = FIGURES_DIR / "fig_convergence_by_format.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 7b: Guided Feature Medians vs AOCC Tiers (Phase 3)
# ===================================================================

def fig_guided_medians(df3, df1):
    """2x5 grid: per-feature bar chart of median behavioral value by format,
    with AOCC tier reference lines from Stage 1 data (bottom-25%, median, top-10%)."""
    bf_df = df3[~df3["failed"]].copy()
    bf_cols = [c for c in bf_df.columns if c.startswith("bf_")]
    bf_df = bf_df.dropna(subset=["fitness"]).reset_index(drop=True)
    if bf_df.empty:
        print("  WARNING: No behavioral feature data -- skipping fig_guided_medians")
        return

    # Compute tier thresholds from Stage 1 data
    df1_valid = df1[~df1["failed"]].copy()
    for col in df1_valid.columns:
        if col.startswith("bm_"):
            df1_valid[col] = pd.to_numeric(df1_valid[col], errors="coerce")
    s1_fitness = df1_valid["fitness"].dropna()
    p25 = s1_fitness.quantile(0.25)
    p50 = s1_fitness.quantile(0.50)
    p90 = s1_fitness.quantile(0.90)

    tiers = {
        "bottom 25%": df1_valid[df1_valid["fitness"] < p25],
        "median":     df1_valid[(df1_valid["fitness"] >= p25) & (df1_valid["fitness"] < p90)],
        "top 10%":    df1_valid[df1_valid["fitness"] >= p90],
    }
    tier_colors = {"bottom 25%": "#d62728", "median": "#7f7f7f", "top 10%": "#2ca02c"}
    tier_styles = {"bottom 25%": "--", "median": ":", "top 10%": "--"}

    fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharey=False)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURES):
        ax = axes_flat[idx]
        # Phase 3 uses bf_ prefix, Phase 1 uses bm_ prefix
        bf_col = f"bf_{feat}"
        bm_col = f"bm_{feat}"
        if bf_col not in bf_df.columns:
            ax.set_visible(False)
            continue

        fmts = formats_for_feature(feat)
        fmt_medians = []
        fmt_stds = []
        for fmt in fmts:
            sub = bf_df[(bf_df["feature"] == feat) & (bf_df["format"] == fmt)][bf_col].dropna()
            fmt_medians.append(sub.median())
            fmt_stds.append(sub.std())

        x_pos = np.arange(len(fmts))
        ax.bar(x_pos, fmt_medians, yerr=fmt_stds,
               color=[FORMAT_COLORS[f] for f in fmts],
               edgecolor="black", linewidth=0.5, width=0.6, zorder=3,
               capsize=4, error_kw={"linewidth": 1.2})

        for tier_name, tier_df in tiers.items():
            if bm_col not in tier_df.columns:
                continue
            tier_med = tier_df[bm_col].dropna().median()
            if pd.isna(tier_med):
                continue
            ax.axhline(y=tier_med, color=tier_colors[tier_name],
                       linestyle=tier_styles[tier_name], linewidth=2, alpha=0.8,
                       zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f[:4] for f in fmts], fontsize=9)
        ax.set_title(FEATURE_SHORT[feat], fontsize=FONT_SIZE_TICK, fontweight="bold")
        if idx % 5 == 0:
            ax.set_ylabel("Median Feature Value")

    # Build combined legend: bar colors for formats + line styles for tiers
    import matplotlib.lines as _mlines
    legend_elements = [
        mpatches.Patch(facecolor=FORMAT_COLORS["neutral"], edgecolor="black",
                       linewidth=0.5, label="Neutral"),
        mpatches.Patch(facecolor=FORMAT_COLORS["directional"], edgecolor="black",
                       linewidth=0.5, label="Directional"),
        mpatches.Patch(facecolor=FORMAT_COLORS["comparative"], edgecolor="black",
                       linewidth=0.5, label="Comparative"),
        _mlines.Line2D([], [], color=tier_colors["bottom 25%"],
                       linestyle=tier_styles["bottom 25%"], linewidth=2, label="Bottom 25% (Stage 1)"),
        _mlines.Line2D([], [], color=tier_colors["median"],
                       linestyle=tier_styles["median"], linewidth=2, label="Median (Stage 1)"),
        _mlines.Line2D([], [], color=tier_colors["top 10%"],
                       linestyle=tier_styles["top 10%"], linewidth=2, label="Top 10% (Stage 1)"),
    ]
    plt.tight_layout()
    fig.legend(handles=legend_elements, loc="upper center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.97), frameon=True)
    plt.subplots_adjust(top=0.90)

    outpath = FIGURES_DIR / "fig_guided_medians.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 8: Steering Shifts (Phase 3)
# ===================================================================

def fig_steering_shifts(df3):
    """2x5 grid showing behavioral shift from neutral per feature per format."""
    # Load FEATURE_DIRECTIONS from experiments.feedback
    try:
        from experiments.feedback import FEATURE_DIRECTIONS
    except ImportError:
        print("  WARNING: Cannot import experiments.feedback -- skipping fig_steering_shifts")
        return

    # Build behavioral feature dataframe from valid candidates
    conditions = []
    for feat in FEATURES:
        for fmt in FORMATS:
            if fmt == "comparative" and feat in COMPARATIVE_EXCLUDE:
                continue
            conditions.append(f"{fmt}-{feat}")

    bf_df = df3[~df3["failed"]].copy()
    bf_cols = [c for c in bf_df.columns if c.startswith("bf_")]
    bf_df = bf_df.dropna(subset=["fitness"]).reset_index(drop=True)

    if bf_df.empty:
        print("  WARNING: No behavioral feature data -- skipping fig_steering_shifts")
        return

    # Compute correlations to determine desired direction
    correlations = {}
    for feat in FEATURES:
        col = f"bf_{feat}"
        if col not in bf_df.columns:
            continue
        valid = bf_df[[col, "fitness"]].dropna()
        if len(valid) > 3:
            rho, p = stats.spearmanr(valid[col], valid["fitness"])
            corr_dir = "higher" if rho > 0 else "lower"
            advice_dir = FEATURE_DIRECTIONS[feat][0]
            correlations[feat] = {"rho": rho, "corr_dir": corr_dir, "advice_dir": advice_dir}

    # Compute steering results
    steering_results = []
    for feat in FEATURES:
        col = f"bf_{feat}"
        if col not in bf_df.columns or feat not in correlations:
            continue
        desired = correlations[feat]["advice_dir"]
        neutral_med = bf_df[(bf_df["feature"] == feat) & (bf_df["format"] == "neutral")][col].median()

        for fmt in formats_for_feature(feat):
            med = bf_df[(bf_df["feature"] == feat) & (bf_df["format"] == fmt)][col].median()
            diff = med - neutral_med
            if fmt == "neutral":
                correct = "-"
            elif desired == "higher":
                correct = "YES" if diff > 0 else "NO"
            else:
                correct = "YES" if diff < 0 else "NO"
            steering_results.append({
                "feature": feat, "format": fmt, "median": med,
                "diff_vs_neutral": diff, "correct": correct, "desired": desired,
            })

    steer_df = pd.DataFrame(steering_results)
    steer_non_neutral = steer_df[steer_df["format"] != "neutral"]

    # Colors: clear green for correct, muted red for wrong
    COLOR_CORRECT = "#59a14f"
    COLOR_WRONG = "#d62728"

    # 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharey=False)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURES):
        ax = axes_flat[idx]
        col = f"bf_{feat}"
        if col not in bf_df.columns or feat not in correlations:
            ax.set_visible(False)
            continue

        fmts = [f for f in formats_for_feature(feat) if f != "neutral"]
        sub = steer_non_neutral[steer_non_neutral["feature"] == feat].set_index("format")
        if sub.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(fmts))
        diffs = [sub.loc[f, "diff_vs_neutral"] if f in sub.index else 0 for f in fmts]
        corrects = [sub.loc[f, "correct"] if f in sub.index else "?" for f in fmts]
        colors_bar = [COLOR_CORRECT if c == "YES" else COLOR_WRONG
                      for c in corrects]

        bars = ax.bar(x, diffs, color=colors_bar, edgecolor="none", width=0.5)
        for j, (bar, c) in enumerate(zip(bars, corrects)):
            if c == "NO":
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "WRONG", ha="center",
                        va="bottom" if bar.get_height() > 0 else "top",
                        fontsize=FONT_SIZE_LEGEND, color=COLOR_WRONG, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "CORRECT", ha="center",
                        va="bottom" if bar.get_height() > 0 else "top",
                        fontsize=FONT_SIZE_LEGEND, color=COLOR_CORRECT, fontweight="bold")

        ax.axhline(y=0, color="#888888", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f[:4] for f in fmts], fontsize=FONT_SIZE_TICK)
        ax.set_title(FEATURE_SHORT[feat], fontsize=FONT_SIZE_TICK, fontweight="bold")
        if idx % 5 == 0:
            ax.set_ylabel("Shift vs Neutral")

    outpath = FIGURES_DIR / "fig_steering_shifts.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Figure 9: MA-BBOB Instance Selection (2-panel)
# ===================================================================

def fig_mabbob_instances():
    """Per-function weight share and per-group weight share in selected MA-BBOB instances."""
    # weights.csv path: ../BLADE/iohblade/benchmarks/BBOB/mabbob/weights.csv
    if not WEIGHTS_CSV.exists():
        print(f"  WARNING: weights.csv not found at {WEIGHTS_CSV}")
        print("  Skipping fig_mabbob_instances")
        return

    weights = pd.read_csv(WEIGHTS_CSV, index_col=0)
    W = weights.values  # 1000 x 24

    GROUPS = {
        "Separable (f1-f5)": list(range(0, 5)),
        "Low/mod conditioning (f6-f9)": list(range(5, 9)),
        "High cond / unimodal (f10-f14)": list(range(9, 14)),
        "Multimodal adequate (f15-f19)": list(range(14, 19)),
        "Multimodal weak (f20-f24)": list(range(19, 24)),
    }

    # Greedy selection (reproduce notebook logic)
    def score_subset(indices):
        sub = W[indices]
        func_totals = sub.sum(axis=0)
        total = func_totals.sum()
        if total == 0:
            return -9999
        missing = (func_totals == 0).sum()
        group_shares = [func_totals[cols].sum() / total for cols in GROUPS.values()]
        group_dev = np.std(group_shares)
        nonzero = func_totals[func_totals > 0]
        func_cv = np.std(nonzero) / np.mean(nonzero) if len(nonzero) > 0 else 999
        return -missing * 100 - group_dev * 10 - func_cv

    selected = []
    remaining = list(range(1000))
    for step in range(10):
        best_score = -9999
        best_idx = -1
        for idx in remaining:
            s = score_subset(selected + [idx])
            if s > best_score:
                best_score = s
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    selected.sort()

    # Compute shares
    sub = W[selected]
    func_totals = sub.sum(axis=0)
    total = func_totals.sum()
    func_labels = [f"f{i+1}" for i in range(24)]
    ideal = 1.0 / 24
    shares = func_totals / total

    group_map = {}
    for gname, cols in GROUPS.items():
        for c in cols:
            group_map[func_labels[c]] = gname.split(" (")[0]

    group_colors = {
        "Separable": "#4e79a7",
        "Low/mod conditioning": "#f28e2b",
        "High cond / unimodal": "#e15759",
        "Multimodal adequate": "#76b7b2",
        "Multimodal weak": "#59a14f",
    }
    bar_colors = [group_colors[group_map[f]] for f in func_labels]

    # --- 2-panel figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # Panel (a): Per-function weight share
    bars = ax1.bar(func_labels, shares * 100, color=bar_colors,
                   edgecolor="white", linewidth=0.5)
    ax1.axhline(y=ideal * 100, color="#555555", linestyle="--", linewidth=1,
                label=f"Ideal ({ideal*100:.1f}%)")
    ax1.set_ylabel("Share of total weight (%)")
    ax1.set_xlabel("BBOB Function")
    ax1.set_title("(a) Per-function weight share", fontweight="bold")
    ax1.set_ylim(0, max(shares * 100) * 1.3)

    for bar, share in zip(bars, shares):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{share*100:.1f}", ha="center", va="bottom", fontsize=FONT_SIZE_LEGEND)

    patches = [mpatches.Patch(color=c, edgecolor="none", label=g)
               for g, c in group_colors.items()]
    patches.append(mlines.Line2D([], [], color="#555555", linestyle="--",
                                 linewidth=1, label="Ideal (4.2%)"))
    ax1.legend(handles=patches, loc="upper right", fontsize=FONT_SIZE_LEGEND)

    # Panel (b): Per-group weight share
    group_names_short = []
    group_shares_pct = []
    group_bar_colors = []
    ideal_group = 20.0  # each of 5 groups should have 20%
    for gname, cols in GROUPS.items():
        short = gname.split(" (")[0]
        group_names_short.append(short)
        g_share = func_totals[cols].sum() / total * 100
        group_shares_pct.append(g_share)
        group_bar_colors.append(group_colors[short])

    x_g = np.arange(len(group_names_short))
    ax2.bar(x_g, group_shares_pct, color=group_bar_colors, edgecolor="white",
            linewidth=0.5, width=0.6)
    ax2.axhline(y=ideal_group, color="#555555", linestyle="--", linewidth=1,
                label=f"Ideal ({ideal_group:.0f}%)")
    ax2.set_xticks(x_g)
    ax2.set_xticklabels(group_names_short, fontsize=FONT_SIZE_LEGEND, rotation=30, ha="right")
    ax2.set_ylabel("Share of total weight (%)")
    ax2.set_title("(b) Per-group weight share", fontweight="bold")
    ax2.set_ylim(0, max(group_shares_pct) * 1.3)

    for i, pct in enumerate(group_shares_pct):
        ax2.text(i, pct + 0.3, f"{pct:.1f}%", ha="center", va="bottom",
                 fontsize=FONT_SIZE_LEGEND)

    ax2.legend(fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_mabbob_instances.pdf"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Exporting thesis figures to PDF")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    # --- Phase 1 figures ---
    print("\n[Phase 1] Loading data...")
    if RESULTS_PHASE1.exists():
        df1 = load_phase1()
        print(f"  Loaded {len(df1)} evaluations across {df1['model'].nunique()} models")

        print("\n[Phase 1] fig_model_screening.pdf")
        summary = fig_model_screening(df1)

        print("[Phase 1] fig_tsne_behavioral.pdf")
        fig_tsne_behavioral(df1)

        print("[Phase 1] fig_spearman.pdf")
        fig_spearman(df1)

        print("[Phase 1] fig_rf_importance.pdf")
        fig_rf_importance(df1)

        print("[Phase 1] fig_ks_effect.pdf")
        fig_ks_effect(df1)

        print("[Phase 1] fig_failure_modes.pdf")
        fig_failure_modes(df1, summary)

        # spearman_heatmap superseded by fig_spearman above
    else:
        df1 = None
        print(f"  WARNING: {RESULTS_PHASE1} not found -- skipping Phase 1 figures")

    # --- Phase 3 figures ---
    print("\n[Phase 3] Loading data...")
    if RESULTS_PHASE3.exists():
        df3 = load_phase3()
        print(f"  Loaded {len(df3)} evaluations across {df3['condition'].nunique()} conditions")

        print("\n[Phase 3] fig_format_boxplot.pdf")
        best3 = fig_format_boxplot(df3)

        print("[Phase 3] fig_condition_ranking.pdf")
        fig_condition_ranking(df3, best3)

        print("[Phase 3] fig_convergence_by_format.pdf")
        fig_convergence_by_format(df3)

        print("[Phase 3] fig_guided_medians.pdf")
        fig_guided_medians(df3, df1)

        print("[Phase 3] fig_steering_shifts.pdf")
        fig_steering_shifts(df3)
    else:
        print(f"  WARNING: {RESULTS_PHASE3} not found -- skipping Phase 3 figures")

    # --- MA-BBOB figure ---
    print("\n[MA-BBOB] fig_mabbob_instances.pdf")
    fig_mabbob_instances()

    # --- Summary ---
    produced = list(FIGURES_DIR.glob("fig_*.pdf"))
    print("\n" + "=" * 60)
    print(f"Done. Produced {len(produced)} figures:")
    for p in sorted(produced):
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
