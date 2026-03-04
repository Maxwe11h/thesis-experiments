# Model Selection Experiment

## Motivation

The feature-selection experiment (Qwen3 8B) revealed a **76.8% failure rate** across 1,200 candidates. Before running the full feasibility study with multiple runs, we need to find an SLM that produces viable code more reliably. This experiment compares models in the 8-30B parameter range on vanilla LLaMEA to select the best candidate.

## Timing analysis from the feature-selection experiment

Reverse-engineering the conversation timestamps from the previous experiment:

| Metric | Value |
|--------|-------|
| Total wall time (12 conditions) | 33.5 hours |
| LLM inference | 28.9 hours (86.2%) |
| Algorithm evaluation | 4.6 hours (13.8%) |
| Avg LLM time per candidate | ~75 seconds |
| Avg successful eval time | ~60 seconds (50 MA-BBOB runs) |
| Avg failed eval time | ~0 seconds (instant compile/smoke-test failure) |
| Time per MA-BBOB run | ~1.2 seconds |

**Key insight**: LLM inference dominates at 86% of total time. Reducing evaluation instances/seeds saves minimal overall time — the real bottleneck is the LLM's ability to produce valid code.

### Should we reduce from 10 instances × 5 seeds?

| Config | Runs/candidate | Eval time/success | % of total time |
|--------|---------------|-------------------|-----------------|
| 10 × 5 (current) | 50 | ~60s | ~14% |
| 10 × 3 | 30 | ~36s | ~9% |
| 5 × 5 | 25 | ~30s | ~7% |
| 5 × 3 | 15 | ~18s | ~4% |

Reducing to 5 × 3 saves ~10% overall — modest. However, for the model selection experiment specifically, reduced evaluation is fine since we care about failure rate and rough AOCC ordering, not precise scores.

## Design

### Models to compare

| Tag | Ollama model | Params | Rationale |
|-----|-------------|--------|-----------|
| `qwen3-8b` | `qwen3:8b` | 8B | Current baseline (feature-selection experiment) |
| `qwen2.5-coder-7b` | `qwen2.5-coder:7b` | 7B | Code-specialized, may follow API templates better |
| `llama3.1-8b` | `llama3.1:8b` | 8B | Strong general-purpose alternative |
| `qwen3-14b` | `qwen3:14b` | 14B | Larger version of current model |
| `qwen2.5-coder-14b` | `qwen2.5-coder:14b` | 14B | Code-specialized, larger |
| `codestral-22b` | `codestral:22b` | 22B | Mistral's code generation model |
| `qwen2.5-coder-32b` | `qwen2.5-coder:32b` | 32B | Largest code model that fits Q4 on 24GB |

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Condition | Vanilla only | Isolates model quality from feedback effects |
| Candidates per model | 50 | Enough for reliable failure rate + a few successful AOCC |
| Evolution strategy | (1+1)-ES | Same as feasibility study |
| MA-BBOB instances | 10 | Same as feasibility study |
| Seeds per instance | 5 | Same (can reduce for faster turnaround) |
| Dimensionality | 5 | Same |

### Selection criteria (priority order)

1. **Failure rate** — primary metric (weight 3x)
2. **Best AOCC achieved** — can the model produce good algorithms? (weight 2x)
3. **LLM inference speed** — affects total experiment time (weight 1x)
4. **Failure mode distribution** — interface mismatches indicate prompt-following ability

## Running the experiment

### Prerequisites

Each model must be pulled on the Ollama instance before running:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:14b
ollama pull codestral:22b
ollama pull qwen2.5-coder:32b
```

### Running

```bash
conda activate /local/$USER/conda_envs/thesis

# All models sequentially:
python run_model_selection.py all

# Specific models:
python run_model_selection.py qwen3-8b qwen2.5-coder-7b llama3.1-8b

# With reduced eval for faster turnaround:
python run_model_selection.py all --budget 50 --training-instances 5 --eval-seeds 3

# On a second GPU:
OLLAMA_PORT=11435 python run_model_selection.py qwen3-14b codestral-22b qwen2.5-coder-32b

# Custom model not in the predefined list:
python run_model_selection.py deepseek-v2 --custom-model "deepseek-coder-v2:16b"
```

### Parallel strategy (vibranium)

Run smaller models on one GPU and larger models on the other:

```bash
# GPU 0 (port 11434): 8B models
nohup python run_model_selection.py qwen3-8b qwen2.5-coder-7b llama3.1-8b > logs/model_sel_8b.log 2>&1 &

# GPU 1 (port 11435): 14-32B models
OLLAMA_PORT=11435 nohup python run_model_selection.py qwen3-14b qwen2.5-coder-14b codestral-22b qwen2.5-coder-32b > logs/model_sel_large.log 2>&1 &
```

### Estimated time

Assuming similar LLM inference speeds to Qwen3 8B (~75s/candidate on RTX 3090):

| Model | Est. per-candidate | Est. total (50 candidates) |
|-------|-------------------|---------------------------|
| 8B models | ~60-90s | ~1-1.5h each |
| 14B models | ~90-150s | ~1.5-2.5h each |
| 22B models | ~150-240s | ~2.5-4h each |
| 32B models | ~200-360s | ~3-5h each |

Total for all 7 models: ~12-22 hours (split across 2 GPUs: ~6-11h wall time)

## Results directory

```
results_model_selection/
  <model_tag>/
    experimentlog.jsonl
    progress.json
    run-<model_tag>-MA_BBOB-0/
      log.jsonl
      conversationlog.jsonl
```

## Analysis

Use the analysis notebook: `analysis/model_selection_analysis.ipynb`

Produces:
- Summary table (failure rate, best/mean AOCC, timing)
- Failure rate comparison bar chart
- Failure mode breakdown per model
- Timing comparison
- Convergence curves
- Cost-efficiency scatter plot
- Final composite ranking
