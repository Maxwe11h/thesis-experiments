# Feature-Selection Experiment

## Objective

Determine which individual behavioral metrics, when included as feedback to the LLM, improve the quality of automatically designed optimization algorithms compared to AOCC-only feedback.

## Design

The experiment uses LLaMEA with a (1+1)-ES evolutionary strategy to generate optimization algorithms for MA-BBOB (Many-Affine Black-Box Optimization Benchmarking). A single LLM (Qwen3 8B with thinking mode enabled) generates candidate algorithms, which are evaluated across 50 runs (10 MA-BBOB instances x 5 seeds). Performance is measured by AOCC (Area Over the Convergence Curve, 1.0 = best).

### Conditions (12 total)

There are 12 experimental conditions. Each condition uses the same LLM, the same prompts, the same evaluation pipeline, and the same evolutionary strategy. The **only** variable is the feedback string returned to the LLM after each candidate evaluation:

| # | Condition | Feedback content |
|---|-----------|-----------------|
| 1 | `vanilla` (baseline) | AOCC score + std dev |
| 2 | `avg_nearest_neighbor_distance` | AOCC + nearest-neighbor distance metric |
| 3 | `dispersion` | AOCC + coverage dispersion metric |
| 4 | `avg_exploration_pct` | AOCC + exploration percentage metric |
| 5 | `avg_distance_to_best` | AOCC + distance-to-best metric |
| 6 | `intensification_ratio` | AOCC + intensification ratio metric |
| 7 | `avg_exploitation_pct` | AOCC + exploitation percentage metric |
| 8 | `average_convergence_rate` | AOCC + convergence rate metric |
| 9 | `avg_improvement` | AOCC + improvement magnitude metric |
| 10 | `success_rate` | AOCC + success rate metric |
| 11 | `longest_no_improvement_streak` | AOCC + no-improvement streak metric |
| 12 | `last_improvement_fraction` | AOCC + last improvement fraction metric |

No guided mutation (SAGE) is used in any condition. This isolates the effect of metric visibility on LLM-guided algorithm design.

### Feedback format

**Vanilla (baseline):**
```
The algorithm AlgorithmName got an average Area over the convergence curve
(AOCC, 1.0 is the best) score of 0.8765 with standard deviation 0.0432.
```

**Single-feature (e.g. dispersion):**
```
The algorithm AlgorithmName got an average Area over the convergence curve
(AOCC, 1.0 is the best) score of 0.8765 with standard deviation 0.0432.
Behavioral metric — dispersion: 2.5678
  (How well the evaluated points cover the search domain)
```

Feature descriptions are deliberately neutral (no directional guidance like "lower is better") to avoid biasing the LLM before we have empirical evidence of which direction helps.

### Behavioral metrics

The 11 metrics are computed from the full optimization trace of each candidate algorithm and grouped into four categories:

**Exploration & Diversity**
- `avg_nearest_neighbor_distance` — How spread out the evaluated points are from each other
- `dispersion` — How well the evaluated points cover the search domain
- `avg_exploration_pct` — Percentage of search effort spent exploring vs exploiting

**Exploitation & Intensification**
- `avg_distance_to_best` — How far each evaluation is from the current best solution
- `intensification_ratio` — How often the algorithm samples near the best-so-far solution
- `avg_exploitation_pct` — Percentage of search effort spent exploiting vs exploring

**Convergence Progress**
- `average_convergence_rate` — Rate at which the best-so-far fitness error decreases per step
- `avg_improvement` — Average size of improvement when a new best solution is found
- `success_rate` — How often an evaluation improves on the best-so-far solution

**Stagnation & Reliability**
- `longest_no_improvement_streak` — Longest run of consecutive evaluations with no improvement
- `last_improvement_fraction` — How much of the budget has passed since the last improvement

## Configuration

| Parameter | Value |
|-----------|-------|
| LLM | Qwen3 8B (via Ollama, thinking mode enabled) |
| LLaMEA budget | 100 candidates per condition |
| Evolution strategy | (1+1)-ES (1 parent, 1 offspring, elitism) |
| MA-BBOB instances | 10 (indices 0-9) |
| Seeds per instance | 5 |
| Evaluations per candidate | 50 (10 instances x 5 seeds) |
| Dimensionality | 5 |
| Budget per algorithm run | 10,000 (2000 x dim) |
| Search bounds | [-5.0, 5.0]^d |
| Allowed imports | numpy |

## Code structure

```
thesis/
  experiments/
    config.py              # Shared constants (BEHAVIORAL_FEATURES, model, ES params)
    feedback.py            # FEATURE_DESCRIPTIONS + vanilla_feedback + make_single_feature_feedback factory
    mabbob_problem.py      # MaBBOBProblem: evaluation with inner seed loop + behavioral metrics
    run_experiment.py      # CONDITIONS dict (12 entries), make_problem, make_method, run_condition
    trajectory_logger.py   # IOH logger capturing per-evaluation (x, y) traces
  run_conditions.py        # CLI runner for arbitrary condition subsets
  setup_server.sh          # Server setup script for REL Compute nodes
  BLADE/                   # Submodule (fork): experiment infrastructure, LLM clients, behaviour_metrics
  LLaMEA/                  # Submodule: LLaMEA evolutionary algorithm framework
```

## Logged outputs

Each condition produces a `results/<condition_name>/` directory containing:

| File | Contents |
|------|----------|
| `experimentlog.jsonl` | Final solution per run with full metadata |
| `progress.json` | Live progress tracking (poll remotely to monitor) |
| `run-*/log.jsonl` | Per-generation candidate: fitness, code, feedback, metadata (incl. `behavioral_features` dict with all 11 metrics, `aucs` list with all 50 scores) |
| `run-*/conversationlog.jsonl` | Full LLM conversation with timestamps per message |

All 11 behavioral metrics are recorded as solution metadata for every candidate in every condition, regardless of which metric (if any) was shown in feedback. This enables post-hoc analysis across all metrics.

## Server deployment

### Infrastructure

Experiments run on LIACS REL Compute servers. Both are GeForce RTX GPU nodes with Ollama support (no SLURM required).

| Server | GPUs | Role |
|--------|------|------|
| vibranium.liacs.nl | 2x RTX 3090 (24GB each) | 8 conditions (2 parallel groups of 4) |
| duranium.liacs.nl | 6x GTX 980 Ti + 2x Titan X | 4 conditions (1 group) |

### Setup (per server)

```bash
# Connect via REL gateway
ssh ssh.liacs.nl
ssh vibranium   # or duranium

# Clone repo to /local (fast I/O, no quota issues)
git clone --recurse-submodules https://github.com/Maxwe11h/thesis-experiments.git /local/$USER/thesis
cd /local/$USER/thesis

# Run setup (creates conda env on /local, installs deps, verifies)
bash setup_server.sh
```

The setup script:
1. Creates a conda environment (Python 3.11) at `/local/$USER/conda_envs/thesis`
2. Installs BLADE and LLaMEA as editable packages (with `--no-deps` to skip unused heavy dependencies)
3. Installs only the runtime dependencies needed for experiments
4. Verifies Ollama access and that `qwen3:8b` is available
5. Runs a verification import check (12 conditions, 11 features)

### Running the experiment

```bash
conda activate /local/$USER/conda_envs/thesis
cd /local/$USER/thesis
```

**Vibranium — group 1 (4 conditions):**
```bash
nohup python run_conditions.py \
  vanilla avg_nearest_neighbor_distance dispersion avg_exploration_pct \
  > logs/group1.log 2>&1 &
```

**Vibranium — group 2 (4 conditions):**
```bash
nohup python run_conditions.py \
  avg_distance_to_best intensification_ratio avg_exploitation_pct average_convergence_rate \
  > logs/group2.log 2>&1 &
```

**Duranium — group 3 (4 conditions):**
```bash
nohup python run_conditions.py \
  avg_improvement success_rate longest_no_improvement_streak last_improvement_fraction \
  > logs/group3.log 2>&1 &
```

The two vibranium groups run as parallel processes sharing one Ollama instance. Since candidate evaluation (50 algorithm runs, CPU-bound) takes much longer than LLM inference, the processes naturally interleave: while one evaluates a candidate, the other gets its Ollama response.

### Monitoring

```bash
# Watch live output
tail -f /local/$USER/thesis/logs/group1.log

# Check per-condition progress
cat /local/$USER/thesis/results/vanilla/progress.json | python3 -m json.tool

# Check remotely (from local machine, via gateway)
ssh ssh.liacs.nl "ssh vibranium 'cat /local/s3815129/thesis/results/vanilla/progress.json'" | python3 -m json.tool
```

### Retrieving results

Results must be copied off `/local` before the monthly reboot (2nd Sunday of each month, ~23:30).

```bash
# From local machine via the gateway
rsync -avz -e "ssh -J ssh.liacs.nl" vibranium:/local/$USER/thesis/results/ ./results_vibranium/
rsync -avz -e "ssh -J ssh.liacs.nl" duranium:/local/$USER/thesis/results/ ./results_duranium/
```

## Analysis plan

For each of the 11 single-feature conditions, compare final AOCC against the vanilla baseline. Features that consistently improve AOCC are candidates for inclusion in subsequent experiments (combined feedback, guided mutation / SAGE).
