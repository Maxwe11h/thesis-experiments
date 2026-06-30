#!/usr/bin/env python
"""Gap-fill reproducibility script for thesis values that had no backing code.

Several numbers reported in the thesis were computed ad hoc and lived only in the
LaTeX (no notebook or script produced them). This script recomputes each such
"orphan" value from the results data, mirroring the exact loaders/filters used by
the analysis notebooks, and prints it next to the thesis value with a MATCH/DIFF
flag.

Scope (decided in the gap-fill audit): traceability only -- give every reported
results value a code home. Where the recomputed value disagrees with the thesis,
the line is flagged DIFF so the text can be corrected.

Run from repo root with the conda-base interpreter that has sklearn + BLADE:
    /opt/miniconda3/bin/python analysis/reported_stats.py

Covers (with thesis ref):
  ch5:404  Levene combined vs vanilla; Welch one-sided t-tests + Holm; <0.88 counts
  ch5:107  Random Forest OOB MAE
  ch5:277  Stage-1 vs Stage-3 mean-AOCC drift (0.484 / 0.646)
  ch5:240  Format-level Kruskal-Wallis (H, p) and the three format mean AOCCs
  tab:correlation-shift  Stage-1 rho/Bot25/Top10/Gap; Stage-3 Top10
  tab:ast-complexity     rho_AOCC + p over 14,127; d_{D-C} Cliff's delta column
  ch3:234  SampEn full vs subsampled timing (~30 s / ~0.3 s)
  ch3:544  per-candidate eval-time range (logged observation -- not recomputable here)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BLADE"))

RESULTS_PHASE1 = ROOT / "results_phase1"
RESULTS_PHASE3 = ROOT / "results_phase3"
RESULTS_PHASE4 = ROOT / "results_phase4"

# ----------------------------------------------------------------------------- helpers


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def chk(label, computed, thesis, tol, fmt="{:.3f}"):
    """Print one reconciliation line: computed vs thesis with a MATCH/DIFF flag."""
    try:
        diff = abs(float(computed) - float(thesis))
        flag = "MATCH" if diff <= tol else "DIFF "
        cstr = fmt.format(computed)
        tstr = fmt.format(thesis)
    except (TypeError, ValueError):
        flag = "----"
        cstr, tstr = str(computed), str(thesis)
    print(f"  [{flag}] {label:<46s} computed={cstr:>12s}  thesis={tstr:>12s}")


def parse_fitness_neg(val):
    """Phase 1 convention: failures -> -inf."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("-inf")


def parse_fitness_nan(val):
    """Phase 3 convention: failures / -inf -> NaN."""
    try:
        f = float(val)
        return f if not (math.isinf(f) and f < 0) else np.nan
    except (TypeError, ValueError):
        return np.nan


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# ---- the 10 selected behavioural features (correlation-shift table order) ----
# (short label, phase-1 bm_ column, phase-3 bf_ column)
SHIFT_FEATURES = [
    ("avg_impr", "avg_improvement", "avg_improvement"),
    ("intens_ratio", "intensification_ratio", "intensification_ratio"),
    ("plateau_frac", "fitness_plateau_fraction", "fitness_plateau_fraction"),
    ("step_autocorr", "step_size_autocorrelation", "step_size_autocorrelation"),
    ("impr_spatial", "improvement_spatial_correlation", "improvement_spatial_correlation"),
    ("half_conv", "half_convergence_time", "half_convergence_time"),
    ("fit_autocorr", "fitness_autocorrelation", "fitness_autocorrelation"),
    ("x_spread", "x_spread_early", "x_spread_early"),
    ("longest_str", "longest_no_improvement_streak", "longest_no_improvement_streak"),
    ("dim_conv_het", "dimension_convergence_heterogeneity", "dimension_convergence_heterogeneity"),
]

# Thesis values from tab:correlation-shift (ch5). rho, bot25, top10 per stage.
THESIS_SHIFT = {
    #               S1_rho  S1_bot25 S1_top10  S3_rho  S3_top10
    "avg_impr":     (-0.782, 1.752, 0.093, -0.184, 0.078),
    "intens_ratio": (+0.725, 0.000, 0.879, +0.475, 0.873),
    "plateau_frac": (+0.689, 0.000, 0.597, +0.578, 0.597),
    "step_autocorr":(+0.764, 0.130, 0.910, +0.468, 0.903),
    "impr_spatial": (+0.757, 0.087, 0.682, +0.446, 0.689),
    "half_conv":    (-0.611, 0.010, 0.001, -0.640, 0.001),
    "fit_autocorr": (+0.650, 0.003, 0.766, +0.138, 0.738),
    "x_spread":     (-0.570, 2.889, 0.621, -0.410, 0.578),
    "longest_str":  (-0.427, 6282., 5543., +0.495, 6018.),
    "dim_conv_het": (+0.424, 0.001, 0.081, +0.167, 0.089),
}

# AST_COLS in qualitative_tier_comparison order; thesis (rho_AOCC, d_{D-C}) from tab:ast-complexity.
AST_COLS = [
    "total_ast_nodes", "n_np_calls", "n_numeric_constants", "n_assignments",
    "n_try_except", "n_functions", "n_for_loops", "n_while_loops",
    "n_if_branches", "max_loop_depth", "n_augmented_assigns", "n_return_statements",
]
THESIS_AST = {
    "total_ast_nodes":     (+0.48, +0.21),
    "n_np_calls":          (+0.60, +0.24),
    "n_numeric_constants": (+0.66, +0.09),
    "n_assignments":       (+0.40, +0.21),
    "n_try_except":        (+0.48, +0.08),
    "n_functions":         (+0.23, +0.04),
    "n_for_loops":         (+0.06, +0.11),
    "n_while_loops":       (-0.15, +0.17),
    "n_if_branches":       (+0.16, +0.09),
    "max_loop_depth":      (+0.16, +0.15),
    "n_augmented_assigns": (+0.47, -0.05),
    "n_return_statements": (+0.03, +0.12),
}

PHASE3_FEATURES = [f[2] for f in SHIFT_FEATURES]
FORMATS = ["neutral", "directional", "comparative"]
COMPARATIVE_EXCLUDE = {"longest_no_improvement_streak"}


# ============================================================ Stage 4 (results_phase4)


def stage4():
    hdr("STAGE 4  (results_phase4)  -- Levene, Welch+Holm, <0.88 counts   [ch5:399-404]")
    conds = ["vanilla", "neutral", "sage", "combined_neutral"]
    seeds = range(10)
    rows = []
    for cond in conds:
        for seed in seeds:
            fp = RESULTS_PHASE4 / cond / f"seed-{seed}" / "summary.csv"
            if not fp.exists():
                continue
            d = pd.read_csv(fp)
            d["condition"] = cond
            d["seed"] = seed
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["AOCC_valid"] = df["AOCC"].where(df["run_status"] == "success")
    df = df.sort_values(["condition", "seed", "generation"]).reset_index(drop=True)
    df["best_so_far"] = df.groupby(["condition", "seed"])["AOCC_valid"].cummax()
    final = (df.groupby(["condition", "seed"])
               .agg(final_best=("best_so_far", "last")).reset_index())

    vals = {c: final[final.condition == c].final_best.values for c in conds}

    # Means / stds (already backed, shown for context)
    thesis_mean = {"vanilla": 0.915, "neutral": 0.931, "combined_neutral": 0.937, "sage": 0.938}
    print("  per-condition final best-so-far AOCC (mean +/- std, ddof=1):")
    for c in conds:
        chk(f"{c} mean", vals[c].mean(), thesis_mean[c], 0.003)

    # <0.88 counts  [ch5:398]
    print("  seeds finishing below 0.88:")
    thesis_below = {"vanilla": 4, "neutral": 1, "sage": 1, "combined_neutral": 0}
    for c in conds:
        chk(f"{c} count<0.88", int((vals[c] < 0.88).sum()), thesis_below[c], 0, "{:.0f}")

    # Welch one-sided t-tests vs vanilla + Holm  [ch5:404]
    print("  Welch one-sided t-test (greater) vs vanilla, raw p:")
    thesis_p = {"combined_neutral": 0.088, "sage": 0.126, "neutral": 0.203}
    feedback = ["combined_neutral", "sage", "neutral"]
    raw = []
    for c in feedback:
        t, p = stats.ttest_ind(vals[c], vals["vanilla"], equal_var=False, alternative="greater")
        raw.append(p)
        chk(f"{c} vs vanilla", p, thesis_p[c], 0.005)
    adj = holm(raw)
    print("  Holm-adjusted (across the 3 pairs):")
    for c, a in zip(feedback, adj):
        print(f"         {c:<28s} p_holm={a:.3f}  {'(<0.05)' if a < 0.05 else '(n.s.)'}")
    print(f"         -> thesis claim 'no test reaches alpha=0.05': "
          f"{'CONFIRMED' if (adj >= 0.05).all() else 'CONTRADICTED'}")

    # Levene combined vs vanilla  [ch5:404]
    print("  Levene equal-variance test, combined vs vanilla:")
    T, p = stats.levene(vals["combined_neutral"], vals["vanilla"])
    chk("Levene T", T, 5.57, 0.05, "{:.2f}")
    chk("Levene p", p, 0.030, 0.003)


# ============================================================ Stage 1 (results_phase1)


def load_phase1():
    MODELS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "rnj-1-8b",
              "devstral-small-2-24b", "olmo3-7b", "olmo3-32b", "granite4-3b",
              "gemini-3-pro", "gemini-3-flash"]
    rows = []
    for model in MODELS:
        for seed in range(5):
            seed_dir = RESULTS_PHASE1 / model / f"seed-{seed}"
            run_dirs = sorted(seed_dir.glob("run-*"))
            if not run_dirs:
                continue
            log_file = run_dirs[-1] / "log.jsonl"  # latest run (mirrors notebook)
            if not log_file.exists():
                continue
            with open(log_file) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    fit = parse_fitness_neg(e.get("fitness"))
                    meta = e.get("metadata", {}) or {}
                    bf = meta.get("behavioral_features", {}) or {}
                    failed = math.isinf(fit) or math.isnan(fit)
                    row = {"model": model, "seed": seed, "fitness": fit, "failed": failed}
                    for k, v in bf.items():
                        col = "fitness_autocorrelation" if k == "fitness_autocorrelation_lag1" else k
                        row[f"bm_{col}"] = v
                    rows.append(row)
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.startswith("bm_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[~df["failed"]].copy()
    bm_all = [c for c in valid.columns if c.startswith("bm_")]
    valid = valid.dropna(subset=bm_all, how="all").reset_index(drop=True)
    # drop metrics that are NaN for >50% of candidates (mirrors notebook -> ~32 features)
    nan_frac = valid[bm_all].isna().mean()
    bm_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    return df, valid, bm_cols


def stage1_rf(valid, bm_cols):
    hdr("STAGE 1  Random Forest OOB R^2 + MAE   [ch5:107]")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    rf_data = valid[bm_cols + ["fitness"]].dropna()
    X, y = rf_data[bm_cols].values, rf_data["fitness"].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"  trained on {len(rf_data)} candidates x {len(bm_cols)} features")
    chk("RF OOB R^2", rf.oob_score_, 0.962, 0.003)
    oob = rf.oob_prediction_
    mask = ~np.isnan(oob)
    mae = mean_absolute_error(y[mask], oob[mask])
    chk("RF OOB MAE", mae, 0.024, 0.003)


def stage1_corr_shift(valid):
    hdr("STAGE 1  correlation-shift columns (rho / Bot25 / Top10)   [tab:correlation-shift]")
    fitness = valid["fitness"]
    q25 = fitness.quantile(0.25)
    q90 = fitness.quantile(0.90)
    bot = valid[valid["fitness"] <= q25]
    top = valid[valid["fitness"] >= q90]
    print(f"  Stage-1 valid candidates n = {len(valid)}  (thesis table header: n=3,086)")
    print(f"  {'feature':<14s} {'rho c/th':>16s}  {'Bot25 c/th':>18s}  {'Top10 c/th':>18s}")
    for short, bm, _ in SHIFT_FEATURES:
        col = f"bm_{bm}"
        if col not in valid.columns:
            print(f"  {short:<14s}  (column missing)")
            continue
        s1_rho_t, s1_bot_t, s1_top_t = THESIS_SHIFT[short][:3]
        sub = valid[[col, "fitness"]].dropna()
        rho, _ = stats.spearmanr(sub[col], sub["fitness"])
        bot_med = bot[col].dropna().median()
        top_med = top[col].dropna().median()
        big = abs(s1_bot_t) > 10  # integer-scale feature (streak)
        bfmt = "{:.0f}" if big else "{:.3f}"
        f_rho = "MATCH" if abs(rho - s1_rho_t) <= 0.02 else "DIFF "
        print(f"  {short:<14s} [{f_rho}] {rho:+6.3f}/{s1_rho_t:+6.3f}   "
              f"{bfmt.format(bot_med):>8s}/{bfmt.format(s1_bot_t):<8s}  "
              f"{bfmt.format(top_med):>8s}/{bfmt.format(s1_top_t):<8s}")


# ============================================================ Stage 3 (results_phase3)


def load_phase3():
    """Mirror phase3_feedback_analysis load_all() + bf_df build."""
    conditions = []
    for feat in PHASE3_FEATURES:
        for fmt in FORMATS:
            if fmt == "comparative" and feat in COMPARATIVE_EXCLUDE:
                continue
            conditions.append((f"{fmt}-{feat}", fmt, feat))
    fit_rows, bf_rows = [], []
    for cond, fmt, feat in conditions:
        for seed in range(5):
            seed_dir = RESULTS_PHASE3 / cond / f"seed-{seed}"
            run_dirs = sorted(seed_dir.glob("run-*"))
            if not run_dirs:
                continue
            log_file = run_dirs[0] / "log.jsonl"
            if not log_file.exists():
                continue
            with open(log_file) as f:
                for i, line in enumerate(f):
                    e = json.loads(line.strip())
                    fitness = parse_fitness_nan(e.get("fitness"))
                    fit_rows.append({"condition": cond, "format": fmt, "feature": feat,
                                     "seed": seed, "fitness": fitness})
                    if np.isnan(fitness):
                        continue
                    bf = e.get("metadata", {}).get("behavioral_features", {})
                    if not bf:
                        continue
                    row = {"format": fmt, "feature": feat, "seed": seed, "fitness": fitness}
                    for bk, bv in bf.items():
                        row[f"bf_{bk}"] = pd.to_numeric(bv, errors="coerce")
                    bf_rows.append(row)
    return pd.DataFrame(fit_rows), pd.DataFrame(bf_rows)


def stage3_drift(df_fit_p3, valid_p1):
    hdr("DRIFT  Stage-1 vs Stage-3 mean AOCC   [ch5:277]")
    s1_mean = valid_p1["fitness"].mean()
    s3_valid = df_fit_p3["fitness"].dropna()
    s3_mean = s3_valid.mean()
    chk("Stage-1 mean AOCC (all valid)", s1_mean, 0.484, 0.01)
    chk("Stage-3 mean AOCC (all valid)", s3_mean, 0.646, 0.01)


def stage3_format_kw(df_fit_p3):
    hdr("FORMAT RANKING  Kruskal-Wallis + format means   [ch5:240]")
    # best AOCC per (condition, seed), then per-(format,feature) mean over seeds
    best = (df_fit_p3.groupby(["condition", "format", "feature", "seed"])["fitness"]
                    .max().reset_index(name="best_aocc"))
    per_ff = (best.groupby(["format", "feature"])["best_aocc"].mean()
                  .reset_index(name="ff_mean"))
    groups = [per_ff[per_ff.format == f]["ff_mean"].values for f in FORMATS]
    print(f"  per-(format,feature) mean AOCCs: "
          + ", ".join(f"{f}={len(g)}" for f, g in zip(FORMATS, groups)))
    H, p = stats.kruskal(*groups)
    chk("Kruskal-Wallis H", H, 11.5, 0.3, "{:.2f}")
    chk("Kruskal-Wallis p", p, 0.003, 0.002)
    thesis_fmt_mean = {"neutral": 0.855, "directional": 0.827, "comparative": 0.784}
    for f, g in zip(FORMATS, groups):
        chk(f"{f} mean AOCC", g.mean(), thesis_fmt_mean[f], 0.01)


def stage3_top10(bf_df):
    hdr("STAGE 3  Top10 reference values   [tab:correlation-shift]")
    fitness = bf_df["fitness"]
    q90 = fitness.quantile(0.90)
    top = bf_df[bf_df["fitness"] >= q90]
    print(f"  Stage-3 valid w/ features n = {len(bf_df)}; top-10% n = {len(top)}")
    for short, _, bf in SHIFT_FEATURES:
        col = f"bf_{bf}"
        if col not in bf_df.columns:
            print(f"  {short:<14s}  (column missing)")
            continue
        s3_top_t = THESIS_SHIFT[short][4]
        med = top[col].dropna().median()
        big = abs(s3_top_t) > 10
        bfmt = "{:.0f}" if big else "{:.3f}"
        tol = 50 if big else 0.02
        flag = "MATCH" if abs(med - s3_top_t) <= tol else "DIFF "
        print(f"  [{flag}] {short:<14s} Top10 computed={bfmt.format(med):>8s}  "
              f"thesis={bfmt.format(s3_top_t):>8s}")


# ============================================================ AST (results_phase3 code)


def ast_metrics(code):
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    m = dict(n_functions=0, n_classes=0, n_for_loops=0, n_while_loops=0,
             n_if_branches=0, n_try_except=0, max_loop_depth=0,
             n_numeric_constants=0, n_string_constants=0, n_np_calls=0,
             n_assignments=0, n_augmented_assigns=0, n_return_statements=0,
             total_ast_nodes=0)

    def walk(node, depth=0):
        m["total_ast_nodes"] += 1
        if isinstance(node, ast.FunctionDef): m["n_functions"] += 1
        elif isinstance(node, ast.ClassDef): m["n_classes"] += 1
        elif isinstance(node, ast.For):
            m["n_for_loops"] += 1; nd = depth + 1
            m["max_loop_depth"] = max(m["max_loop_depth"], nd)
            for c in ast.iter_child_nodes(node): walk(c, nd)
            return
        elif isinstance(node, ast.While):
            m["n_while_loops"] += 1; nd = depth + 1
            m["max_loop_depth"] = max(m["max_loop_depth"], nd)
            for c in ast.iter_child_nodes(node): walk(c, nd)
            return
        elif isinstance(node, ast.If): m["n_if_branches"] += 1
        elif isinstance(node, ast.Try): m["n_try_except"] += 1
        elif isinstance(node, ast.Assign): m["n_assignments"] += 1
        elif isinstance(node, ast.AugAssign): m["n_augmented_assigns"] += 1
        elif isinstance(node, ast.Return): m["n_return_statements"] += 1
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)): m["n_numeric_constants"] += 1
            elif isinstance(node.value, str): m["n_string_constants"] += 1
        elif isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"):
                m["n_np_calls"] += 1
        for c in ast.iter_child_nodes(node):
            walk(c, depth)

    walk(tree)
    return m


def cliffs_d_fast(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined, method="average")
    u_x = ranks[:n_x].sum() - n_x * (n_x + 1) / 2
    return 2 * u_x / (n_x * n_y) - 1


def ast_table():
    hdr("AST COMPLEXITY  rho_AOCC over 14,127 + d_{D-C} column   [tab:ast-complexity]")
    # Mirror qualitative_tier_comparison: load Stage-3 with code, all-valid tier.
    rows = []
    for feat in PHASE3_FEATURES:
        for fmt in FORMATS:
            if fmt == "comparative" and feat in COMPARATIVE_EXCLUDE:
                continue
            cond = f"{fmt}-{feat}"
            for seed in range(5):
                seed_dir = RESULTS_PHASE3 / cond / f"seed-{seed}"
                runs = sorted(seed_dir.glob("run-*"))
                if not runs:
                    continue
                log = runs[0] / "log.jsonl"
                if not log.exists():
                    continue
                with open(log) as fh:
                    for line in fh:
                        e = json.loads(line.strip())
                        f_ = parse_fitness_nan(e.get("fitness"))
                        rows.append({"format": fmt, "feature": feat, "seed": seed,
                                     "fitness": f_, "failed": np.isnan(f_),
                                     "code": e.get("code", "")})
    df = pd.DataFrame(rows)
    valid = df[~df["failed"]].copy()
    parsed = []
    for _, r in valid.iterrows():
        m = ast_metrics(r["code"])
        if m is None:
            continue
        m["format"] = r["format"]
        m["fitness"] = r["fitness"]
        parsed.append(m)
    ast_df = pd.DataFrame(parsed)
    print(f"  all-valid tier: parsed {len(ast_df)} / {len(valid)} algorithms "
          f"(thesis: 14,127)")
    print(f"  {'metric':<22s} {'rho c/th':>16s}   {'d_DC c/th':>16s}")
    for col in AST_COLS:
        rho_t, dDC_t = THESIS_AST[col]
        sub = ast_df[[col, "fitness"]].dropna()
        rho, _ = stats.spearmanr(sub[col], sub["fitness"])
        d_vals = [ast_df[ast_df.format == f][col].values for f in FORMATS]
        d_DC = cliffs_d_fast(d_vals[1], d_vals[2])  # directional vs comparative
        f_rho = "MATCH" if abs(rho - rho_t) <= 0.03 else "DIFF "
        f_d = "MATCH" if abs(d_DC - dDC_t) <= 0.03 else "DIFF "
        print(f"  {col:<22s} [{f_rho}] {rho:+5.2f}/{rho_t:+5.2f}    "
              f"[{f_d}] {d_DC:+5.2f}/{dDC_t:+5.2f}")


# ============================================================ SampEn timing  [ch3:234]


def sampen_timing():
    hdr("SampEn TIMING  full (step=1) vs subsampled (step=10)   [ch3:234]")
    from iohblade.behaviour_metrics import fitness_sample_entropy
    T, d = 10_000, 5
    rng = np.random.default_rng(42)
    x = np.zeros(d)
    positions, fitnesses = [x.copy()], []
    for t in range(T):
        if t == 0:
            fitnesses.append(float(np.sum(x ** 2)))
        else:
            x_new = np.clip(x + rng.normal(0, 0.5, d), -5, 5)
            f_new = float(np.sum(x_new ** 2)) + rng.normal(0, 0.1)
            if f_new < fitnesses[-1]:
                x = x_new
            positions.append(x.copy())
            fitnesses.append(f_new if f_new < fitnesses[-1] else fitnesses[-1] + rng.normal(0, 0.01))
    pos = np.array(positions[:T])
    df = pd.DataFrame({"evaluations": np.arange(T), "raw_y": fitnesses[:T],
                       **{f"x{j}": pos[:, j] for j in range(d)}})

    def timeit(fn, n=5):
        fn()  # warmup
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    t_full = timeit(lambda: fitness_sample_entropy(df, m=2, subsample_step=1))
    t_sub = timeit(lambda: fitness_sample_entropy(df, m=2, subsample_step=10))
    print(f"  full  (N=10,000, step=1):  {t_full:8.3f} s   (thesis text: ~30 s)")
    print(f"  subsampled (N=1,000, step=10): {t_sub*1000:7.1f} ms   (thesis text: ~0.3 s = 300 ms)")
    print(f"  speed-up factor: {t_full / t_sub:.0f}x  (O(N^2) predicts ~100x for 10x subsampling)")
    print("  NOTE: ch3:234 absolute values to be reconciled against these measurements.")


def eval_time_note():
    hdr("EVAL-TIME RANGE  30-120 s / candidate, 1-3 h / run   [ch3:544]  (ambiguous)")
    print("  This is a wall-clock observation from run logs, not a statistic recomputable")
    print("  here without re-running 100-candidate jobs. Related instrumentation lives in")
    print("  experiments/benchmark_eval_overhead.py (RandomSearch, 3 instances, ~1-3.2 s/eval).")
    print("  FLAGGED for text softening rather than a fabricated number.")


# ============================================================ main


def main():
    print("Gap-fill reconciliation: recomputed value vs thesis value (MATCH within tol / DIFF)")
    stage4()
    df_p1, valid_p1, bm_cols = load_phase1()
    stage1_rf(valid_p1, bm_cols)
    stage1_corr_shift(valid_p1)
    df_fit_p3, bf_df = load_phase3()
    stage3_drift(df_fit_p3, valid_p1)
    stage3_format_kw(df_fit_p3)
    stage3_top10(bf_df)
    ast_table()
    sampen_timing()
    eval_time_note()
    print("\nDone.")


if __name__ == "__main__":
    main()
