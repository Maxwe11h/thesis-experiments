# Phase 1: LLM Screening Experiment

Screen multiple LLMs for their baseline algorithm-discovery capability on the MA-BBOB benchmark using LLaMEA's evolutionary loop with a (1+1)-ES strategy.

## Quick start

```bash
# List available models
python run_phase1.py --list

# Run all 5 seeds for one model
python run_phase1.py qwen3.5-4b

# Sanity check (fast, ~5 min)
python run_phase1.py qwen3.5-4b --sanity
```

## Experiment design

| Parameter | Value |
|-----------|-------|
| Population strategy | (1+1) — 1 parent, 1 offspring, elitism |
| Mutation mix | 90% refinement, 10% exploration (new algorithm) |
| Independent runs per LLM | 5 (seeds 0–4) |
| Initial population | 1 hand-crafted algorithm (RandomSearch, shared across all models/seeds) |
| MA-BBOB instances | 10 training instances |
| Evaluation seeds | 5 per instance (50 total evaluations per candidate) |
| Dimensionality | 5 |
| Budget factor | 2000 (→ 10 000 function evaluations per algorithm run) |
| LLaMEA budget | 100 candidates per run (1 initial + 99 generations) |

### Models screened

**Local (Ollama):**
- Qwen 3.5: 4B, 9B, 27B
- rnj-1: 8B
- Devstral-small-2: 24B
- OLMo 3: 7B, 30B
- Granite 4: 3B

**API (Gemini):**
- Gemini 3 Pro
- Gemini 3 Flash

## Dependencies

Ensure the conda environment is set up per `setup_server.sh`. The Phase 1 experiment uses the same dependencies as the feature-selection experiment, plus:

```bash
# For Gemini API models
pip install google-genai
```

## Configuring Ollama models

Pull each model on the server before running:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull qwen3.5:27b
ollama pull rnj-1:8b
ollama pull devstral-small-2:24b
ollama pull olmo3:7b
ollama pull olmo3:30b
ollama pull granite4:3b
```

Verify availability:

```bash
ollama list
```

If model names differ from what's in the registry, either:
1. Edit `experiments/phase1_config.py` → `CANDIDATE_MODELS`, or
2. Use `--custom-ollama` at the CLI:
   ```bash
   python run_phase1.py my-tag --custom-ollama "actual-model-name:tag"
   ```

### Multi-GPU setup

For servers with multiple GPUs, run a second Ollama instance:

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 \
  OLLAMA_MODELS=/local/$USER/.ollama/models \
  ollama serve > logs/ollama_gpu1.log 2>&1 &

# Pull models into the second instance
OLLAMA_HOST=http://localhost:11435 ollama pull qwen3.5:27b
```

Then use `OLLAMA_PORT=11435` when running:

```bash
OLLAMA_PORT=11435 python run_phase1.py qwen3.5-27b
```

## CLI reference

```
python run_phase1.py [models...] [options]

positional arguments:
  models                Model tag(s) or 'all'

options:
  --list                List registered models
  --seeds SEED [SEED...] Run seeds (default: 0 1 2 3 4)
  --budget N            Candidates per run (default: 100)
  --port PORT           Ollama port (default: OLLAMA_PORT env or 11434)
  --eval-seeds N        Inner evaluation seeds per instance (default: 5)
  --training-instances N  Number of MA-BBOB instances (default: 10)
  --no-worker-pool      Evaluate via subprocess-per-call
  --custom-ollama MODEL Use an arbitrary Ollama model
  --custom-gemini MODEL Use an arbitrary Gemini model
  --sanity              Quick validation mode
  --summarise           Generate summary CSVs for existing results
  --results-dir PATH    Base results directory (default: results_phase1)
```

## Running on the server (nohup)

### Single model, all seeds

```bash
conda activate /local/$USER/conda_envs/thesis
cd /local/$USER/thesis

nohup python run_phase1.py qwen3.5-4b \
  > logs/phase1-qwen3.5-4b.log 2>&1 &
```

### Parallel execution across GPUs

**GPU 0 (port 11434):**

```bash
nohup python run_phase1.py qwen3.5-4b qwen3.5-9b rnj-1-8b granite4-3b \
  > logs/phase1-gpu0.log 2>&1 &
```

**GPU 1 (port 11435):**

```bash
OLLAMA_PORT=11435 nohup python run_phase1.py qwen3.5-27b devstral-small-2-24b olmo3-7b olmo3-30b \
  > logs/phase1-gpu1.log 2>&1 &
```

**Gemini API (no GPU needed, can run anywhere):**

```bash
GOOGLE_API_KEY=your-key-here nohup python run_phase1.py gemini-3-pro gemini-3-flash \
  > logs/phase1-gemini.log 2>&1 &
```

### Per-seed nohup (maximum parallelism)

```bash
for seed in 0 1 2 3 4; do
  nohup python run_phase1.py qwen3.5-4b --seeds $seed \
    > logs/phase1-qwen3.5-4b-seed${seed}.log 2>&1 &
done
```

## Monitoring

```bash
# Running processes
jobs

# Live log output
tail -f logs/phase1-gpu0.log

# Per-model progress
for d in results_phase1/*/seed-*/; do
  echo "=== $d ==="
  cat "$d/progress.json" 2>/dev/null | python3 -m json.tool
  echo
done

# GPU status
nvidia-smi
```

## Sanity tests

### Quick validation (no GPU inference)

```bash
pytest tests/test_phase1_sanity.py -v
```

Tests initial population generation, config consistency, and summary extraction.

### Integration test (requires Ollama)

```bash
pytest tests/test_phase1_sanity.py -v -k integration
```

Runs 1 seed, 2 instances, 1 eval seed with a real model.

### CLI sanity mode

```bash
python run_phase1.py qwen3.5-4b --sanity
```

Runs: 2 instances, 1 eval seed, 1 run seed, 10 candidates. Takes ~10 minutes instead of hours.

## Output structure

```
results_phase1/
  qwen3.5-4b/
    seed-0/
      progress.json                     # live progress tracker
      experimentlog.jsonl               # final experiment summary
      summary.csv                       # generated post-run
      run-qwen3.5-4b-MA_BBOB-0/
        log.jsonl                       # per-candidate: fitness, code, feedback, metadata
        conversationlog.jsonl           # full LLM conversation history
        stdout.log                      # captured stdout
        stderr.log                      # captured stderr
    seed-1/
      ...
    seed-2/
      ...
  qwen3.5-9b/
    ...
  gemini-3-pro/
    ...
```

### Summary CSV fields

| Field | Description |
|-------|-------------|
| `model_name` | Model tag |
| `seed` | Run seed |
| `generation` | Generation number |
| `algorithm_name` | Class name of generated algorithm |
| `AOCC` | Area Over Convergence Curve (1.0 = best) |
| `final_best_value` | Same as AOCC for successful candidates |
| `run_status` | `success` or `failure` |
| `error` | Error message if failed |
| `n_aucs` | Number of AUC values computed |
| `bm_*` | Behavioral metric values (if computed) |

## Performance considerations

- **LLM inference dominates**: ~75 sec/candidate for 8B model on RTX 3090
- **Evaluation**: ~60 sec/candidate (50 MA-BBOB runs at 10k budget each)
- **Total per run**: ~100 candidates × 135 sec ≈ 3.75 hours
- **Total per model**: 5 runs × 3.75h ≈ 18.75 hours (sequential seeds)
- **Parallel seeds**: Running seeds on separate processes halves wall time

### Memory

- 4B models: ~4 GB VRAM
- 8B–9B models: ~8 GB VRAM
- 24B–30B models: ~20+ GB VRAM, need full RTX 3090
- API models: no local VRAM needed

### Recommended GPU assignment

| GPU | Models |
|-----|--------|
| RTX 3090 (GPU 0) | Small models: 3B, 4B, 7B, 8B, 9B |
| RTX 3090 (GPU 1) | Large models: 24B, 27B, 30B |
| CPU / API | Gemini 3 Pro, Gemini 3 Flash |
