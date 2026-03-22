# Thesis Plan: Behavioural Feedback for LLM-Driven Metaheuristic Design

## Working Title

**Behavioural Feedback for LLM-Driven Metaheuristic Design: Can Trajectory Features Guide the Evolution of Optimization Algorithms?**

## Research Question

Does incorporating trajectory-derived behavioural features into the evolutionary feedback loop of LLM-driven algorithm design improve performance, and does the framing of that feedback matter?

## Narrative Arc

LLMs can design optimization algorithms (LLaMEA) → those algorithms have rich behavioral profiles (Behaviour Space) → code features can guide the search (SAGE) → can behavioral features do the same? → Yes, but only as neutral context -- prescriptive steering is counterproductive.

---

## Chapter Structure

### 1. Introduction (~4 pages)

**Voice**: Direct, clear, sets up the "why" immediately.

- 1.1 **Context**: Automated algorithm design and the LLM paradigm shift. Metaheuristics are everywhere but designing them is manual and slow. LLMs changed this -- LLaMEA generates algorithms from scratch in code space.
- 1.2 **The feedback gap**: Current LLM-driven AAD uses scalar fitness (AOCC) as the sole feedback signal. This compresses the entire optimization trajectory into one number, discarding information about *how* the algorithm searches. Behavioral features capture exploration-exploitation balance, convergence dynamics, and stagnation patterns.
- 1.3 **Research question** (stated formally)
- 1.4 **Contributions**:
  - Extended behavioral feature set (7 novel trajectory metrics on top of 11 existing BLADE metrics)
  - Multi-model LLM screening establishing small language model viability for automated algorithm design
  - Systematic evaluation of three feedback formats across 10 features, revealing that neutral observation outperforms prescriptive guidance
- 1.5 **Thesis outline** (brief paragraph)

**Key citations**: vanStein2024LLaMEA, vanStein2025Behaviour, vanStein2026LLaMEASAGE

---

### 2. Background & Related Work (~8-10 pages)

- 2.1 **Metaheuristic Optimization** (~1.5 pages)
  - The optimization problem (Eq. 1 from LLaMEA: minimize F: S → R)
  - Key algorithm families: ES, DE, CMA-ES, PSO
  - Why the design space is vast -- hundreds of published methods, most never systematically benchmarked
  - **Citations**: eiben2015introduction, back1996evolutionary, hansen2001cmaes, storn1997de

- 2.2 **Automated Algorithm Design** (~2 pages)
  - Rice's algorithm selection problem → ParamILS/irace (configuration) → modular frameworks (modular CMA-ES, modular DE) → the shift from fixed-structure to free-form generation
  - Programming by Optimisation, Genetic Improvement
  - **Citations**: rice1976algorithmselection, hutter2009paramils, smithmiles2009meta, hoos2012pbo, petke2018gi, vanRijn2016modular, vermetten2023modde

- 2.3 **LLMs for Algorithm Design** (~2.5 pages)
  - Three classes (following LLaMEA's taxonomy):
    - Prompt optimization (EvoPrompt, APE)
    - LLMs as EC (EvoLLM -- evolving distributions)
    - Code generation (FunSearch, EoH/AEL, LLaMEA)
  - LLaMEA in detail: the (1+1)-ES loop, initialization, mutation, selection, feedback
  - Recent competitors: MCTS-AHD, LHNS, STOP
  - **Citations**: guo2024evoprompt, zhou2023ape, romeraparedes2024funsearch, liu2024eoh, liu2023ael, vanStein2024LLaMEA, zheng2025mctsahd, xie2025lhns, zelikman2024stop

- 2.4 **Benchmarking & Evaluation** (~1.5 pages)
  - BBOB and its limitations → MA-BBOB (many-affine combinations for diversity)
  - COCO platform, IOHexperimenter/IOHanalyzer
  - AOCC metric: captures anytime performance, not just final solution quality
  - BLADE framework: standardized evaluation for LLM-driven AAD
  - **Citations**: hansen2021coco, vermetten2023mabbob, deNobel2024iohexperimenter, wang2022iohanalyzer, vanStein2025BLADE

- 2.5 **Behavioral Analysis of Optimization Algorithms** (~2 pages)
  - Fitness landscape analysis (ELA, flacco)
  - Trajectory-based analysis: STNs (graph-based visualization), DynamoRep (population dynamics), Kostovska's trajectory features
  - The Behaviour Space paper as direct precursor: defines 11 behavioral metrics for LLM-generated algorithms, uses them for post-hoc analysis but *not* as feedback
  - **Citations**: mersmann2011ela, malan2013landscape, munoz2015information, kerschke2019flacco, ochoa2021stn, cenikj2023dynamorep, kostovska2022trajectory, cenikj2026features, vanStein2025Behaviour

- 2.6 **Structured Feedback in AAD** (~1 page)
  - LLaMEA-SAGE: code features (AST graph metrics, cyclomatic complexity) → surrogate model → SHAP → natural-language guidance. Feeds back *structural* properties.
  - This thesis: behavioral features from execution trajectories. Feeds back *dynamic* properties. Complementary signal from a different level of abstraction.
  - The key question: can an LLM translate high-level behavioral descriptions into the right code changes?
  - **Citations**: vanStein2026LLaMEASAGE, lundberg2017shap

---

### 3. Behavioral Feature Design (~6-8 pages)

This is a contribution chapter. Present all 32 features with formulas, interpretations, and sources.

- 3.1 **Design principles**
  - Features should be cheap to compute (not bottleneck the loop)
  - Features should capture distinct aspects of search dynamics
  - Features should be interpretable in natural language (for LLM feedback)
  - Features should discriminate good from bad algorithms

- 3.2 **Existing BLADE metrics** (11 features, ~2 pages)
  - Exploration & diversity: avg_nearest_neighbor_distance, dispersion, avg_exploration_pct
  - Exploitation & intensification: avg_distance_to_best, intensification_ratio, avg_exploitation_pct
  - Convergence progress: average_convergence_rate, avg_improvement, success_rate
  - Stagnation: longest_no_improvement_streak, last_improvement_fraction
  - **Citations**: vanStein2025Behaviour, vanStein2025BLADE

- 3.3 **Step-size & movement dynamics** (4 features, ~1 page)
  - step_size_mean, step_size_std, step_size_trend, directional_persistence
  - Standard trajectory mechanics from meta-feature literature
  - **Citations**: cenikj2026features

- 3.4 **Information-theoretic features** (6 features, ~2 pages)
  - fitness_sample_entropy, fitness_permutation_entropy, fitness_autocorrelation, fitness_lempel_ziv_complexity, fitness_hurst_exponent, fitness_dfa_alpha
  - Novel application of established time-series measures to optimization fitness trajectories
  - **Citations**: richman2000sampen, bandt2002permutation, weinberger1990fitness, lempel1976complexity, hurst1951storage, peng1994dfa

- 3.5 **Adapted population dynamics** (7 features, ~1 page)
  - x_spread_early, x_spread_late, spread_ratio, centroid_drift, f_range_early, f_range_late, f_range_ratio
  - Adapted from DynamoRep for (1+1)-ES where population size is 1
  - **Citations**: cenikj2023dynamorep

- 3.6 **Novel features** (6 features, ~2 pages)
  - step_size_autocorrelation, fitness_plateau_fraction, half_convergence_time, improvement_spatial_correlation, improvement_burstiness, dimension_convergence_heterogeneity
  - Each gets a paragraph: formula, what it measures, why it's new, what it reveals
  - **Citations**: barabasi2005bursts (burstiness inspiration)

- 3.7 **Computational cost analysis** (~0.5 pages)
  - Table showing per-trajectory cost by category
  - Total overhead <2 minutes per candidate (vs. 30-120s for MA-BBOB evaluation itself)

---

### 4. Methodology (~6 pages)

- 4.1 **LLaMEA framework** (~1.5 pages)
  - The (1+1)-ES evolutionary loop (Algorithm 1 pseudocode)
  - Initialization: shared initial population (RandomSearch + SimpleES)
  - Mutation: 90% refine / 10% explore mix
  - Selection: elitist (1+1)-ES
  - Feedback prompt template
  - **Citations**: vanStein2024LLaMEA

- 4.2 **Benchmark configuration** (~1 page)
  - MA-BBOB: 10 training instances, 5 evaluation seeds, d=5, budget factor 2000
  - AOCC metric definition (Eq. from BLADE)
  - Instance selection rationale
  - **Citations**: vermetten2023mabbob, vanStein2025BLADE

- 4.3 **Stage 1: LLM screening** (~1 page)
  - 10 models: 8 Ollama (Qwen3.5 4B/9B/27B, RNJ-1 8B, Devstral 24B, OLMo 3 7B/32B, Granite4 3B) + 2 Gemini API (3-Pro, 3-Flash)
  - 5 independent seeds per model, 100 candidates per run
  - Composite ranking: failure rate, best AOCC, inference efficiency
  - **Citations**: qwen3, gemini, mistral7b, ollama, kwon2023vllm

- 4.4 **Stage 2: Feature analysis & selection** (~1 page)
  - Dataset: all valid candidates from Stage 1 across 10 models × 5 seeds
  - Three ranking methods: Spearman ρ with AOCC, RF permutation importance, KS effect size
  - Borda-count consensus ranking
  - Correlation-based de-duplication (|ρ| > 0.8)
  - Result: 10 selected features from 32 candidates
  - **Citations**: spearman1904proof, breiman2001randomforests, lundberg2017shap

- 4.5 **Stage 3: Feedback format screening** (~1.5 pages)
  - Model: gemini-3-flash (Phase 1 winner on efficiency)
  - Design: 10 features × 3 formats = 29 conditions (one excluded), 5 seeds each
  - Three feedback formats with examples:
    - **Neutral**: reports value + definition
    - **Directional**: adds "higher/lower is better"
    - **Comparative**: adds distance-aware comparison to top-10% reference
  - Comparative reference value construction (top-10% median, bottom-25% anchor, normalized distance tiers)
  - Why no comparative for longest_no_improvement_streak (U-shaped relationship)

- 4.6 **Statistical methodology** (~0.5 pages)
  - Kruskal-Wallis for overall format comparison
  - Mann-Whitney U for pairwise comparisons
  - Cliff's delta for effect sizes
  - AOCC-matched band analysis (controlling for algorithm quality)
  - Bonferroni correction for multiple comparisons
  - **Citations**: kruskal1952ranks, mann1947test, cliff1993dominance

---

### 5. Results (~12-15 pages)

The main body of the thesis. Heavy on figures from notebooks.

- 5.1 **Stage 1: LLM Screening** (~3 pages)
  - 5.1.1 **Model ranking table** (failure rate, best AOCC, tokens/sec, composite rank)
  - 5.1.2 **Failure mode analysis** (interface mismatch, code generation failure, runtime errors -- show the LLM struggles with API compliance, not algorithmic design)
  - 5.1.3 **Key insight**: model size ≠ code quality for optimization; efficiency matters for AAD budgets. Gemini-3-flash selected for Phase 3.
  - **Figures**: model ranking bar chart, failure mode breakdown, convergence curves across models
  - **Notebooks**: `analysis/phase1_model_ranking.ipynb`

- 5.2 **Stage 2: Feature Analysis** (~2-3 pages)
  - 5.2.1 **Feature-AOCC correlations** (Spearman heatmap across all 10 models, showing consistency)
  - 5.2.2 **Importance ranking** (RF + SHAP + Borda consensus table)
  - 5.2.3 **Redundancy pruning** (inter-feature correlation matrix, which features were dropped and why)
  - 5.2.4 **Selected features** (final top-10 table)
  - **Figures**: Spearman heatmap, importance bar plot, correlation matrix, parallel coordinate plot of feature profiles
  - **Notebooks**: `analysis/phase1_behavior_analysis.ipynb`

- 5.3 **Stage 3: Behavioral Feedback** (~7-10 pages) -- THE MAIN CONTRIBUTION
  - 5.3.1 **Overall format comparison** (~2 pages)
    - Neutral > Directional > Comparative (mean AOCC: 0.869 > 0.840 > 0.788)
    - Kruskal-Wallis p=0.003
    - **Figure**: boxplot of best AOCC by format, convergence curves by format
  - 5.3.2 **Per-feature analysis** (~2 pages)
    - Full 29-condition ranking table
    - Neutral dominates across 4/5 top features
    - Best conditions: neutral-intensification_ratio (0.907), neutral-fitness_plateau_fraction (0.891)
    - **Figure**: heatmap of condition × AOCC, grouped by feature and format
  - 5.3.3 **The steering paradox** (~2 pages)
    - Directional advice is empirically correct (verified against Spearman correlations)
    - Yet directional/comparative feedback pushes features the wrong way in aggregate
    - Key table: feature direction advice vs actual behavioral shift
    - **Figure**: behavioral shift violin plots by format
  - 5.3.4 **AOCC-matched analysis** (~2 pages)
    - Within quality bands (low/mid/high/elite), 16/19 feature × band combinations show significant behavioral differences between formats
    - But steering accuracy is ~50% (coin flip) for both directional and comparative
    - Feature-by-feature breakdown: which features are steerable (fitness_plateau_fraction) vs resistant (step_size_autocorrelation)
    - **Figure**: AOCC-matched behavioral shift heatmap
  - 5.3.5 **Neutral algorithms have naturally good profiles** (~1 page)
    - Performance tier analysis: neutral algorithms sit between mid-50% and top-25% behavioral profiles without explicit steering
    - **Table**: feature values by performance tier vs by format
  - **Notebooks**: `analysis/phase3_feedback_analysis.ipynb`

---

### 6. Discussion (~4-5 pages)

- 6.1 **Why neutral works best: the indirect mapping problem** (~1.5 pages)
  - The Maillard reaction analogy: telling the LLM "increase step_size_autocorrelation" is like telling a chef "increase the Maillard reaction"
  - The mapping from code changes to emergent behavioral metrics is too indirect for prescriptive guidance
  - Neutral feedback works because: (1) gives richer context without prescribing action, (2) evolutionary selection pressure implicitly teaches good behavioral patterns, (3) avoids multi-objective tension between AOCC and feature targets
  - **Key takeaway**: selection mechanism should remain the primary behavioral driver; feedback serves an informational role

- 6.2 **Behavioral vs. structural feedback** (~1 page)
  - LLaMEA-SAGE uses code features (proximal: directly in LLM's output space)
  - This thesis uses behavioral features (distal: emergent from execution)
  - The abstraction gap: code features can give actionable advice ("reduce cyclomatic complexity"), behavioral features describe emergent consequences that the LLM cannot directly control
  - Implication: behavioral and structural feedback may be complementary -- code features for fine-grained guidance, behavioral features for situational context

- 6.3 **The steerability spectrum** (~0.5 pages)
  - fitness_plateau_fraction is the most steerable feature (comparative consistently pushes right across all AOCC bands)
  - step_size_autocorrelation is the least steerable (deeply emergent, resists direct guidance)
  - Hypothesis: steerability correlates with how directly the feature maps to identifiable code patterns

- 6.4 **Implications for AAD feedback design** (~1 page)
  - Practical recommendation: include behavioral features as neutral context, not as optimization targets
  - For researchers building on LLaMEA/BLADE: enrich the feedback signal without constraining the search
  - Open question: could adaptive feedback (switching from neutral in early generations to directional in late generations) combine the benefits?

- 6.5 **Limitations** (~1 page)
  - 5 seeds per condition (adequate power for large effects, limited for subtle ones)
  - Single model in Stage 3 (gemini-3-flash; results may differ for other LLMs)
  - Single dimension (d=5) and fixed budget (2000×d evaluations)
  - No Stage 4 full benchmark (scope reduced due to compute constraints)
  - Behavioral features computed post-hoc on trajectories; no online adaptation

---

### 7. Conclusion & Future Work (~2 pages)

- 7.1 **Summary**: Behavioral features provide useful observational context that slightly improves or maintains LLM-driven algorithm design when presented neutrally. Prescriptive feedback degrades performance despite the LLM clearly responding to it.
- 7.2 **The paradox restated**: The LLM changes what it produces in response to behavioral feedback (16/19 AOCC-matched comparisons significant), but it cannot reliably translate high-level behavioral objectives into the right code changes. Behavioral steering is not systematically wrong but unpredictable.
- 7.3 **Future work**:
  - Stage 4 full benchmark across dimensions and function groups
  - Multi-feature feedback (combining complementary behavioral features)
  - Adaptive feedback format (neutral early, targeted late)
  - Combined behavioral + structural feedback (SAGE + behavioral)
  - Larger population strategies ((μ+λ)-ES with behavioral diversity maintenance)
  - Cross-model validation (do steering patterns generalize across LLMs?)

---

## Appendices

- **A.** Prompt templates (task prompt, feedback format examples, mutation prompts)
- **B.** Complete condition-level results tables (all 29 conditions, all 5 seeds)
- **C.** Per-instance performance heatmaps

(No separate feature definitions appendix -- all feature details go in Chapter 3.)

---

## Figures Needed (from notebooks or to create)

| # | Description | Source | Chapter |
|---|-------------|--------|---------|
| 1 | LLaMEA evolutionary loop diagram | Redraw from LLaMEA paper | 2 or 4 |
| 2 | Model ranking bar chart (Phase 1) | phase1_model_ranking.ipynb | 5.1 |
| 3 | Failure mode breakdown (pie/bar) | To create from logs | 5.1 |
| 4 | Convergence curves by model | phase1_model_ranking.ipynb | 5.1 |
| 5 | Spearman correlation heatmap (features × models) | phase1_behavior_analysis.ipynb | 5.2 |
| 6 | Feature importance bar plot (RF/SHAP/Borda) | phase1_behavior_analysis.ipynb | 5.2 |
| 7 | Inter-feature correlation matrix | phase1_behavior_analysis.ipynb | 5.2 |
| 8 | Parallel coordinate plot (behavioral profiles) | phase1_behavior_analysis.ipynb | 5.2 |
| 9 | Overall format comparison boxplot | phase3_feedback_analysis.ipynb | 5.3 |
| 10 | Convergence curves by format | To create | 5.3 |
| 11 | Full condition ranking heatmap | phase3_feedback_analysis.ipynb | 5.3 |
| 12 | Behavioral shift violin plots | To create | 5.3 |
| 13 | AOCC-matched steering heatmap | To create | 5.3 |
| 14 | Performance tier × format table/figure | To create | 5.3 |
| 15 | Feature steerability comparison | To create | 6.3 |

---

## Writing Order

1. **Chapter 4 (Methodology)** -- most straightforward, establishes experimental setup
2. **Chapter 3 (Feature Design)** -- contribution chapter, formulas and interpretations
3. **Chapter 5 (Results)** -- data-driven, relies on notebooks
4. **Chapter 2 (Background)** -- literature review, cite-heavy
5. **Chapter 6 (Discussion)** -- synthesizes results
6. **Chapter 1 (Introduction)** -- written last with full picture
7. **Chapter 7 (Conclusion)** -- brief wrap-up

---

## Models Used (for correct citation)

### Phase 1 (LLM Screening) -- 10 models:
| Model | Provider | Size | Cite |
|-------|----------|------|------|
| Qwen3.5 4B | Qwen/Alibaba | 4B | qwen3 |
| Qwen3.5 9B | Qwen/Alibaba | 9B | qwen3 |
| Qwen3.5 27B | Qwen/Alibaba | 27B | qwen3 |
| RNJ-1 8B | EssentialAI | 8B | model card |
| Devstral Small 2 24B | Mistral | 24B | mistral7b (base family) |
| OLMo 3 7B | AllenAI | 7B | OLMo 2 tech report |
| OLMo 3 32B Think | AllenAI | 32B | OLMo 2 tech report |
| Granite 4 Micro 3B | IBM | 3B | Granite 3 tech report |
| Gemini 3 Pro | Google | API | gemini, gemini15 |
| Gemini 3 Flash | Google | API | gemini, gemini15 |

### Phase 3 (Feedback Screening) -- 1 model:
| Model | Provider | Cite |
|-------|----------|------|
| Gemini 3 Flash | Google | gemini, gemini15 |

### Earlier experiments (feature selection, discarded):
| Model | Provider | Cite |
|-------|----------|------|
| Qwen3 8B | Qwen/Alibaba | qwen3 |

### Infrastructure:
- Ollama (local inference): cite as software
- vLLM (high-throughput serving): kwon2023vllm
