# Stage 4 analysis design

Date: 2026-05-03
Author: Max Harell
Status: Draft for review

## Goal

Produce the analysis required to write `ch5_results.tex` §5.4 (Stage 4 results). Stage 4 answers the *overarching* RQ — "Can trajectory-based behavioural feedback guide LLMs to design better optimisation algorithms?" — by comparing four feedback conditions (vanilla / neutral / sage / combined_neutral) on the 20-function MA-BBOB Stage 4 set, 500 candidates × 10 seeds, gemini-3-flash with thinking disabled.

This document fixes the section structure, the figures and tables each subsection produces, the source of every artefact (existing notebook vs. new analysis vs. new experiment), and the implementation order.

## Non-goals

- Writing the LaTeX prose — that happens after this analysis is run and signed off.
- Re-running Stage 4 itself.
- Investigating the directional condition (10 seeds were also collected but it falls outside the four-way head-to-head and outside the methodology's described conditions).

## Headline narrative (locked)

> All three feedback extensions improve over vanilla, with separation emerging as early as gen 100. SAGE alone produces the highest mean AOCC; behavioural-only feedback (neutral) is a smaller boost. The headline practical finding is that combining behavioural and structural feedback collapses variance (std drops from ~0.04 to 0.009) and dominates per-instance robustness (11 / 20 wins) — i.e. the most reliable algorithm generator, with mean AOCC statistically indistinguishable from SAGE.

This framing is honest about behavioural-alone being middle-of-pack, leans on the proximal/distal dichotomy from ch1, and lets the practical finding (combined's reliability) carry the section.

## Anchor numbers (from existing `phase4_analysis.ipynb`, executed 2026-05-03)

| Condition | Mean ± std | Median | 95% bootstrap CI |
|---|---|---|---|
| vanilla | 0.915 ± 0.048 | 0.946 | [0.887, 0.942] |
| neutral | 0.931 ± 0.036 | 0.942 | [0.907, 0.946] |
| sage | 0.938 ± 0.038 | 0.952 | [0.913, 0.952] |
| combined | 0.937 ± **0.009** | 0.937 | [0.932, 0.942] |

- Kruskal–Wallis across the four conditions: H = 3.93, p = 0.27 (n.s. overall).
- Strongest pairwise contrast: sage vs combined p = 0.038 raw / 0.226 Holm (Cliff's δ = +0.56, but driven by sage's bimodal seed distribution rather than a mean shift).
- Per-instance wins (out of 20): combined 11, sage 6, neutral 3, vanilla 0.
- Failure rates: neutral 13.9%, vanilla 16.8%, combined 17.6%, sage 21.0% (KW p = 0.38).
- Stage 1 Gemini-Flash failure rate was 2.0% — Stage 4 is roughly 8× higher and warrants explanation.
- Stage 3 → Stage 4 neutral-vs-vanilla delta: +0.045 → +0.016 (effect shrinks as features are bundled and budget grows).

## Section structure (ch5 §5.4)

Six subsections, five figures, five tables, plus one new sub-experiment.

### §5.4.1 Aggregate condition performance

Establishes the headline ranking and the early-gen separation point.

- **Fig.** Best-so-far AOCC over 500 candidates, mean ± std across 10 seeds, four conditions overlaid. Annotate the gen ≈ 100 separation.
- **Table.** Per-condition mean AOCC, std, median, 95 % bootstrap CI; KW test on finals; pairwise MWU with Cliff's δ and Holm correction.
- **Source.** Existing `phase4_analysis.ipynb`. Cells already produce both artefacts; replace `std` ribbons with SEM / 95 % CI per revision-plan note 34.

### §5.4.2 Variance and per-instance robustness (headline finding)

The featured subsection. Carries the "combined wins on reliability" story.

- **Fig.** 2-panel: (a) boxplot of final best AOCC per condition with strip-plot overlay; (b) per-instance heatmap (4 conditions × 20 instances) of mean best-found AOCC.
- **Sub-claim 1.** Variance collapses for combined: std at gen 500 is 0.009 vs ~0.04 elsewhere.
- **Sub-claim 2.** Combined wins 11 / 20 instances, sage 6, neutral 3, vanilla 0.
- **Source.** Existing notebook produces both panels. Confirm the 4 × 20 matrix is built from `log.jsonl` per-instance AUCs.

### §5.4.3 Failure-rate analysis (medium depth, per Q6=b)

Sanity-checks the spike from 2 % (Stage 1) → ~17 % (Stage 4). Quantifies *where in the run* failures occur (the brittle-code-accumulates hypothesis) and breaks failures down by root-cause category. The Gemini-API-change hypothesis is mentioned only as a prose caveat, not tested empirically.

- **Fig.** 2-panel:
  - (a) Per-condition failure rate, per-seed boxplot.
  - (b) Failure rate over evolutionary time, 100-candidate bins, 4 conditions overlaid (gen 0–99, 100–199, 200–299, 300–399, 400–499). Tests the brittle-code hypothesis.
- **Table.** Failure-mode categorisation per condition. Categories adopted from `analysis/FAILURE_ANALYSIS.md`: interface mismatch (wrong `__init__`/`__call__` signature), code-generation failure (no valid Python class produced), runtime error (correct structure, bug in logic). Reusing Stage 1's categories preserves stage-to-stage comparability.
- **Source.** New analysis on existing logs.
  - Failure-mode classification — extend the Stage 1 categoriser to Stage 4's `experimentlog.jsonl` and `summary.csv` `error` strings.
  - Generation-binned rate — `summary.csv` has `generation` and `run_status`; trivial groupby.
- **Caveat to flag in prose.** `thinking_budget = 0` was set for Stage 4 to control variance; Stage 1 used default thinking. The Gemini API platform was also undergoing changes during the experiment window. Both are plausible drivers of the elevated failure rate but distinguishing them empirically would require re-running candidates with thinking enabled, which is out of scope. We mention these confounds in prose without testing them.

### §5.4.4 Behavioural profiles per condition

Quantifies how much each condition's feedback actually moved behaviour, and surfaces a Stage 4 echo of the Stage 3 "steering ≠ performance" lesson.

- **Fig.** Five violin plots (one per tracked neutral feature) split by condition, with Stage 1 top-10 % reference dashed.
- **Table.** Per-condition median feature value vs Stage 1 top-10 % / median / bottom-25 % references. Final column: percentage of candidates that moved feature value toward the Stage 1 top-10 % reference, relative to vanilla. Operationalises revision-plan note 9 ("quantify steering success").
- **Sub-finding.** Combined produces the most "stereotypically high-AOCC" profile (intensification_ratio 0.900, dim_conv_heterogeneity 0.018, plateau_fraction 0.594) without being the top-mean condition. Echoes the Stage 3 finding that pushing single behavioural values toward top-tier references is not a sufficient condition for higher AOCC.
- **Source.** Existing notebook produces medians; the % steered metric is a small extension.

### §5.4.5 The four best algorithms (code-side identity)

Addresses the user-flagged point that LLaMEA's *true* goal is generating the best algorithm. Reads the four winners and characterises what the LLM produced under each feedback regime.

- **Selection.** For each condition, take the seed whose final best-so-far AOCC is highest. Pull that algorithm's source code from the seed's `run-*` directory.
- **Table.** 4 rows × {algorithmic family, named components, lines of code, max nesting depth, distinctive structural feature, mean AOCC on Stage 4 set, comments}.
- **Categorisation.** Manual reading of code into a metaheuristic family (CMA-ES variant / DE / PSO / Nelder–Mead / hybrid / novel). Use an LLM-assisted summarisation pass for a structured first draft, then a manual sanity-check.
- **Sub-claim to test.** Does each feedback condition push the LLM toward a recognisable family? Or do all four winners belong to the same family with parameter differences? The answer informs the discussion's interpretation of behavioural-vs-structural feedback.
- **Source.** New analysis. Source files exist in `results_phase4/<cond>/seed-<N>/run-*/log.jsonl` (algorithm code is in the conversation log; check `experimentlog.jsonl` for the structured list).

### §5.4.6 Full-suite generalisation experiment (new experiment)

Tests whether the four winners are good *optimisers* or just good *Phase-4-instance-fitters*. Pushes the dimension-generalisation question by running at 5D / 10D / 20D, since Stage 4 trained at 5D only.

- **Setup.**
  - **Algorithms:** the four winners from §5.4.5, plus CMA-ES via IOHexperimenter as an external baseline. (Whether to add a second baseline — e.g. modular CMA-ES, or a previously published LLaMEA winner — is deferred until the run is staged.)
  - **Test set:** all 1,000 MA-BBOB functions, 5 instances each. Note in prose that 20 of the 1,000 are Stage 4's training set; we report the comparison across the whole suite without splitting.
  - **Dimensions:** 5, 10, 20.
  - **Budget:** 2,000 × d FEs per run, matching BBOB convention. Total ≈ 1.75 × 10⁹ FEs across all algorithms × dimensions, hours of CPU time on a multi-core node since BBOB evaluations are microsecond-scale.
- **Fig.** Bar chart of mean AOCC per (algorithm × dimension), grouped by BBOB function category (separable / low-mod conditioning / high-uni / multimodal-adequate / multimodal-weak). Shows where each algorithm wins and where it falls off.
- **Table.** Per-dimension mean AOCC for each of the five algorithms, plus delta vs. CMA-ES.
- **Open question to surface.** Did the 5D-trained algorithms hardcode dim-specific constants? If their performance collapses at 20D while CMA-ES holds, that's a finding about LLM-generated code's dimension-portability.

## Implementation order

1. **First analysis pass — §§5.4.1 through 5.4.5.** All on existing data. Builds the Stage 4 narrative end-to-end. Output: a fully populated draft notebook plus all figures saved to `analysis/figs_phase4/`. This is the longest part because §5.4.3 and §5.4.5 require new code (failure categoriser, code-identity classifier).
2. **Stage 4.6 sub-experiment.** Stage and run on saronite (or another available LIACS node). Output feeds §5.4.6 only; everything else is independent.
3. **Final polish.** Replace any std bands with SEM (revision-plan note 34); reconcile failure-rate numbers between this section and any earlier-stage references.

## Open decisions (deferred)

- §5.4.6 baselines beyond CMA-ES — revisit when staging the run.
- Whether to include the directional condition anywhere — the data exists but it is not part of Stage 4's four-way design. Default is to omit.
- Statistical-test choice for the headline (revision-plan note 8) — current notebook uses MWU + Cliff's δ + Holm; bootstrap CIs are also computed. Both feature in §5.4.1; we may swap one out at write-up time.

## Things this design does *not* address

- Discussion section (`ch6_discussion.tex`) — out of scope; will reference the findings produced here.
- Conclusion update — out of scope.
- Code rename `fitness_*` → `objective_*` (revision-plan post-thesis item).

## File-level checklist

Inputs (read-only):
- `results_phase4/{vanilla,neutral,sage,combined_neutral}/seed-{0..9}/summary.csv`
- `results_phase4/{vanilla,neutral,sage,combined_neutral}/seed-{0..9}/experimentlog.jsonl`
- `results_phase4/{vanilla,neutral,sage,combined_neutral}/seed-{0..9}/run-*/log.jsonl`
- `results_phase4/{vanilla,neutral,sage,combined_neutral}/seed-{0..9}/run-*/conversationlog.jsonl`

Outputs:
- Updated `analysis/phase4_analysis.ipynb` (extended sections for failure-mode + steering quantification + best-algorithm identity).
- New notebook `analysis/phase4_full_suite.ipynb` (sub-experiment) — exists by §5.4.6.
- Figures saved to `analysis/figs_phase4/`.
- Tables saved as CSV alongside figures for direct LaTeX import.
- One new run script under `experiments/` for the §5.4.6 sub-experiment, plus a SLURM submission file.
