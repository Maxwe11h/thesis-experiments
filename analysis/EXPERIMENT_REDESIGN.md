# Experiment Redesign: Addressing Variance and Statistical Validity

## 1. The Problem

The feature-selection experiment ran each of the 12 conditions **exactly once** (one evolutionary run of 100 candidates). This means:

- We cannot distinguish whether a condition's performance is due to the feedback metric or due to **random variation** in the LLM's generation process.
- The (1+1)-ES is inherently stochastic: different runs produce different candidate algorithms, different failure sequences, and different final-best scores.
- Statistical tests applied within a single run (Mann-Whitney on candidates, Wilcoxon on AUC scores) measure **within-run** variation but say nothing about **between-run** reproducibility.
- A single lucky or unlucky early generation can dominate the entire trajectory of a (1+1)-ES run.

**Conclusion**: The current experiment cannot establish that any condition reliably improves over vanilla. It is a screening experiment — useful for identifying candidates for further study, but not for drawing causal conclusions.

## 2. What We Need: Between-Run Variance

The missing quantity is **σ_run** — the standard deviation of final-best AOCC across independent runs of the same condition. This number determines everything:

- If σ_run is small (e.g., 0.03), then the observed difference between avg_improvement (0.705) and vanilla (0.416) is overwhelmingly significant even with 2-3 runs.
- If σ_run is large (e.g., 0.15), then even the largest observed difference (+0.289) could plausibly arise from chance, and we'd need many runs to confirm it.

**We do not currently know σ_run.** It cannot be inferred from the within-run candidate variance, because the candidates are not independent (they share an evolutionary trajectory).

## 3. How Many Runs Do We Need?

### 3.1 Power Analysis Framework

For a two-sample t-test comparing final-best AOCC between a condition and vanilla, each with N independent runs:

- **α = 0.05** (significance level)
- **1 - β = 0.80** (80% power — probability of detecting a real effect)
- **Δ** = true AOCC difference between condition and vanilla
- **σ** = σ_run (between-run standard deviation)
- **d = Δ / σ** (Cohen's d effect size)

Required N per condition (two-sample, two-sided):

| d (effect size) | Required N per group | Meaning |
|----------------|---------------------|---------|
| 0.8 (large) | 26 | Δ = 0.8σ |
| 1.0 | 17 | Δ = σ |
| 1.2 | 12 | Δ = 1.2σ |
| 1.5 | 9 | Δ = 1.5σ |
| 2.0 | 6 | Δ = 2σ |
| 3.0 | 4 | Δ = 3σ |

### 3.2 What If We Use Non-Parametric Tests?

Mann-Whitney U is more appropriate for small samples with unknown distributions. It requires approximately 15% more samples than the t-test for equivalent power, so multiply the numbers above by ~1.15.

### 3.3 Practical Scenarios

Using observed differences from the current experiment as our best guess for Δ:

**Scenario A: σ_run = 0.05 (optimistic — low run-to-run variance)**

| Condition | Observed Δ | d = Δ/σ | Runs needed |
|-----------|-----------|---------|-------------|
| avg_improvement | +0.289 | 5.8 | 3 |
| avg_exploitation_pct | +0.244 | 4.9 | 3 |
| longest_no_improvement_streak | +0.165 | 3.3 | 4 |
| dispersion | +0.060 | 1.2 | 12 |

**Scenario B: σ_run = 0.10 (moderate variance)**

| Condition | Observed Δ | d = Δ/σ | Runs needed |
|-----------|-----------|---------|-------------|
| avg_improvement | +0.289 | 2.9 | 4 |
| avg_exploitation_pct | +0.244 | 2.4 | 5 |
| longest_no_improvement_streak | +0.165 | 1.7 | 8 |
| dispersion | +0.060 | 0.6 | 45 |

**Scenario C: σ_run = 0.15 (pessimistic — high run-to-run variance)**

| Condition | Observed Δ | d = Δ/σ | Runs needed |
|-----------|-----------|---------|-------------|
| avg_improvement | +0.289 | 1.9 | 6 |
| avg_exploitation_pct | +0.244 | 1.6 | 8 |
| longest_no_improvement_streak | +0.165 | 1.1 | 14 |
| dispersion | +0.060 | 0.4 | 100+ |

### 3.4 Literature Reference

The BLADE framework defaults to **5 independent runs** per experimental condition (`BLADE/iohblade/experiment.py:32`, `seeds` parameter default). The BLADE README example uses `runs=5`. This is the standard in the LLM-driven algorithm design field and is consistent with general metaheuristic benchmarking conventions (5-10 runs).

## 4. Recommended Approach

### Phase 1: Measure σ_run (pilot study)

**Run vanilla 5 times** with different random seeds. This gives us:
- An empirical estimate of σ_run
- A basis for computing the required N for the full experiment
- Takes 5 × (time for one condition) of compute

**How to implement:**
```python
# In run_conditions.py, pass different evolutionary seeds
nohup python run_conditions.py vanilla --seed 0 --results-dir results_pilot/vanilla_s0 &
nohup python run_conditions.py vanilla --seed 1 --results-dir results_pilot/vanilla_s1 &
# ... etc
```

Currently `run_condition()` passes `seeds=[0]` to BLADE's `Experiment`. The seed controls the LLM conversation (through random parent selection in the ES) and inner evaluation randomness. Different seeds should produce meaningfully different evolutionary trajectories.

### Phase 2: Compute Required N

From the 5 vanilla runs, compute:
- σ_run = std([final_best_run_1, final_best_run_2, ..., final_best_run_5])
- For each candidate condition, compute required N using the observed Δ from the screening experiment

### Phase 3: Focused Multi-Run Experiment

Run only the **top candidates** from the screening experiment, plus vanilla. Based on the current results, the strongest candidates are:

| Condition | Screening Δ | Rationale for inclusion |
|-----------|-------------|------------------------|
| vanilla | (baseline) | Required as control |
| avg_improvement | +0.289 | Largest effect, 10/10 instances better |
| avg_exploitation_pct | +0.244 | Second largest, significant Mann-Whitney |
| longest_no_improvement_streak | +0.165 | 10/10 instances, different category |
| dispersion | +0.060 | Significant Mann-Whitney, strongest fitness correlation |

Running 5 conditions instead of 12 reduces compute by 58%.

### Phase 3a: Apply Prompt Fixes First

Before running the multi-run experiment, apply prompt engineering fixes to reduce the failure rate (see `FAILURE_ANALYSIS.md`). This improves the effective sample size per run, making each run more informative and potentially reducing the required N. Validate with a quick 20-candidate pilot on vanilla to confirm the failure rate drops.

## 5. Compute Budget Estimates

Each condition takes approximately T hours on one GPU (based on the original experiment timing). The multi-run experiment would require:

| Scenario | Conditions | Runs each | Total runs | GPUs (parallel) | Time |
|----------|-----------|-----------|------------|-----------------|------|
| Pilot (phase 1) | 1 (vanilla) | 5 | 5 | 2 | ~2.5T hours |
| Conservative (σ=0.15) | 5 | 10 | 50 | 2 | ~25T hours |
| Moderate (σ=0.10) | 5 | 5 | 25 | 2 | ~12.5T hours |
| Optimistic (σ=0.05) | 5 | 5 | 25 | 2 | ~12.5T hours |

With vibranium's 2 GPUs running in parallel and duranium's GPU running a third stream, the moderate scenario takes approximately 8-9T hours of wall-clock time.

## 6. Alternative Designs

### 6.1 Paired Design (Reduces Required N)

Instead of independent runs, use a **paired design**: for each seed, run both vanilla and the treatment condition. Then apply a **paired t-test** or **Wilcoxon signed-rank** on the differences. This controls for seed-level variation and typically requires fewer runs:

| Effect size (d) | Independent N | Paired N (ρ=0.5) | Paired N (ρ=0.8) |
|----------------|---------------|-------------------|-------------------|
| 1.0 | 17 | 9 | 4 |
| 1.5 | 9 | 5 | 3 |
| 2.0 | 6 | 4 | 3 |

**Implementation**: Run `vanilla_seed_0, avg_improvement_seed_0, vanilla_seed_1, avg_improvement_seed_1, ...` ensuring each pair uses the same random seed for the evolutionary process.

### 6.2 Reduced Budget Per Run

Instead of 100 candidates per run, use 50 or even 30. The convergence curves from the screening experiment show that most conditions find their best algorithm within the first 40-50 evaluations. A shorter budget per run allows more independent runs for the same total compute, which is better for statistical validity than fewer long runs.

| Budget per run | Runs for same compute | Statistical power |
|---------------|----------------------|-------------------|
| 100 (current) | N | Baseline |
| 50 | 2N | ~1.4× more power |
| 30 | 3.3N | ~1.8× more power |

### 6.3 Bootstrap Confidence Interval (No Additional Runs)

If additional compute is truly not available, we can construct confidence intervals using the **within-run candidates** via bootstrap resampling of the evolutionary trajectory. This is weaker than independent runs (it doesn't capture run-level variance) but provides some uncertainty quantification beyond a single point estimate. This is what the current notebook already does — but the limitations should be stated clearly.

## 7. Summary of Recommendations

1. **Immediate**: Run vanilla 5× to measure σ_run (pilot study).
2. **Before multi-run experiment**: Apply prompt engineering to reduce failure rate below 50%.
3. **Multi-run experiment**: 5 conditions × N runs (N determined by pilot σ_run estimate, likely 5-10), using paired design with shared seeds.
4. **Consider reduced budget**: 50 candidates per run instead of 100, allowing more independent runs.
5. **Statistical analysis**: Paired t-test or Wilcoxon signed-rank on final-best AOCC across paired runs, with Bonferroni correction for 4 comparisons (α_adj = 0.0125).

srun --partition=L40s_students --gres=gpu:1 --time=00:15:00 bash -c 'export OLLAMA_MODELS=/local/$USER/ollama_models && mkdir -p $OLLAMA_MODELS && /local/$USER/ollama/bin/ollama serve &sleep 5 && /local/$USER/ollama/bin/ollama pull qwen3.5:4b && /local/$USER/ollama/bin/ollama run qwen3.5:4b "Say hello in one word" && kill %1'

export OLLAMA_MODELS=/local/$USER/ollama_models /local/$USER/ollama/bin/ollama serve > /tmp/ollama.log 2>&1 &