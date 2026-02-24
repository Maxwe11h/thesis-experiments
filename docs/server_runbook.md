# Server Runbook

Operational guide for running and monitoring the feature-selection experiment on REL Compute servers.

## Server overview

| Server | GPUs | Conditions | Ollama setup |
|--------|------|------------|-------------|
| vibranium.liacs.nl | 2x RTX 3090 (24GB) | 8 (two parallel groups of 4) | System Ollama on GPU 0 (port 11434) + user Ollama on GPU 1 (port 11435) |
| duranium.liacs.nl | 6x GTX 980 Ti + 2x Titan X | 4 (one group) | System Ollama (port 11434) |

### Condition assignments

| Group | Server | Port | Conditions |
|-------|--------|------|------------|
| 1 | vibranium | 11434 (system) | `vanilla`, `avg_nearest_neighbor_distance`, `dispersion`, `avg_exploration_pct` |
| 2 | vibranium | 11435 (user GPU 1) | `avg_distance_to_best`, `intensification_ratio`, `avg_exploitation_pct`, `average_convergence_rate` |
| 3 | duranium | 11434 (system) | `avg_improvement`, `success_rate`, `longest_no_improvement_streak`, `last_improvement_fraction` |

## First-time setup

### Connecting

```bash
ssh ssh.liacs.nl          # REL gateway (required from outside LIACS network)
ssh vibranium              # or: ssh duranium
```

### Clone and install

```bash
git clone --recurse-submodules https://github.com/Maxwe11h/thesis-experiments.git /local/$USER/thesis
cd /local/$USER/thesis
bash setup_server.sh
```

The setup script creates a conda environment at `/local/$USER/conda_envs/thesis` with Python 3.11, installs BLADE and LLaMEA as editable packages (skipping unused heavy deps like streamlit, mlflow, plotly), and verifies the setup.

### Updating an existing installation

```bash
cd /local/$USER/thesis
git pull --ff-only
git submodule update --init --recursive
conda activate /local/$USER/conda_envs/thesis
pip install --no-deps -e ./LLaMEA -e ./BLADE
```

## Starting experiment runs

Always activate the environment first:

```bash
conda activate /local/$USER/conda_envs/thesis
cd /local/$USER/thesis
```

### Vibranium

**Second Ollama instance on GPU 1** (needed for group 2):

```bash
# Start the server
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 OLLAMA_MODELS=/local/$USER/.ollama/models ollama serve > logs/ollama_gpu1.log 2>&1 &

# Wait and verify it's alive
sleep 5
OLLAMA_HOST=http://localhost:11435 ollama list
```

If the model isn't listed, pull it into the local instance:

```bash
OLLAMA_HOST=http://localhost:11435 ollama pull qwen3:8b
```

**Group 1** (system Ollama, GPU 0, port 11434):

```bash
nohup python run_conditions.py vanilla avg_nearest_neighbor_distance dispersion avg_exploration_pct > logs/group1.log 2>&1 &
```

**Group 2** (user Ollama, GPU 1, port 11435):

```bash
OLLAMA_PORT=11435 nohup python run_conditions.py avg_distance_to_best intensification_ratio avg_exploitation_pct average_convergence_rate > logs/group2.log 2>&1 &
```

### Duranium

**Group 3** (system Ollama, port 11434):

```bash
nohup python run_conditions.py avg_improvement success_rate longest_no_improvement_streak last_improvement_fraction > logs/group3.log 2>&1 &
```

## Monitoring

### On the server

```bash
# Running processes
jobs

# Live log output
tail -f logs/group1.log
tail -f logs/group2.log

# Per-condition progress (evaluations = candidates evaluated so far, out of 100)
cat results/vanilla/progress.json | python3 -m json.tool

# All conditions at once
for d in results/*/; do echo "=== $(basename $d) ==="; cat "$d/progress.json" 2>/dev/null | python3 -m json.tool; echo; done

# GPU status
nvidia-smi
```

### Remotely (from local machine)

```bash
# Via gateway
ssh ssh.liacs.nl "ssh vibranium 'cat /local/s3815129/thesis/results/vanilla/progress.json'" | python3 -m json.tool

# Quick status of all conditions on a server
ssh ssh.liacs.nl "ssh vibranium 'for d in /local/s3815129/thesis/results/*/; do echo \$(basename \$d): \$(python3 -c \"import json; p=json.load(open(\\\"\$d/progress.json\\\")); r=p.get(\\\"runs\\\",[]); print(r[0][\\\"evaluations\\\"] if r else 0)\" 2>/dev/null)/100; done'"
```

### From vibranium to duranium (or vice versa)

```bash
ssh duranium "cat /local/s3815129/thesis/results/avg_improvement/progress.json" | python3 -m json.tool
```

## Log files

### Directory structure

```
/local/$USER/thesis/
  logs/
    group1.log              # stdout/stderr for group 1 process
    group2.log              # stdout/stderr for group 2 process
    ollama_gpu1.log         # second Ollama instance log (vibranium only)
  results/
    <condition_name>/
      progress.json         # live progress: evaluations count, start/end times
      experimentlog.jsonl   # final solution with full metadata
      run-<condition>-MA_BBOB-0/
        log.jsonl           # per-candidate: fitness, code, feedback, behavioral_features, aucs
        conversationlog.jsonl  # full LLM conversation with timestamps per message
        stdout.log          # captured stdout from the run
        stderr.log          # captured stderr from the run
```

### Key fields in progress.json

```json
{
  "start_time": "2026-02-24T20:13:05.182010",
  "end_time": null,
  "current": 0,
  "total": 1,
  "runs": [{
    "method_name": "vanilla",
    "problem_name": "MA_BBOB",
    "seed": 0,
    "budget": 100,
    "evaluations": 6,
    "start_time": "2026-02-24T20:13:27.253510",
    "end_time": null,
    "log_dir": "run-vanilla-MA_BBOB-0"
  }]
}
```

- `evaluations` increments with each candidate evaluated (target: 100)
- `end_time` is set when the condition completes
- `current` / `total` tracks overall run completion

## Retrieving results

Results live on `/local` which is **not backed up** and is wiped on OS reinstalls. Copy results off before the monthly reboot (2nd Sunday, ~23:30).

```bash
# From local machine via the REL gateway
rsync -avz -e "ssh -J ssh.liacs.nl" vibranium.liacs.nl:/local/s3815129/thesis/results/ ./results_vibranium/
rsync -avz -e "ssh -J ssh.liacs.nl" duranium.liacs.nl:/local/s3815129/thesis/results/ ./results_duranium/
```

Or copy to `/data` on the server as intermediate backup:

```bash
cp -r /local/$USER/thesis/results /data/$USER/thesis_results_$(date +%Y%m%d)
```

## Troubleshooting

### Condition shows "✅" instantly without running

BLADE detected a leftover `results/<condition>/` directory from a previous (possibly killed) run. Delete it and restart:

```bash
rm -rf results/<condition_name>
# then re-run the nohup command for that group
```

### Second Ollama instance won't start

Check the log:

```bash
cat logs/ollama_gpu1.log
```

Common issues:
- **Permission denied on model path**: Use a local model directory (`/local/$USER/.ollama/models`) and pull the model into the second instance with `OLLAMA_HOST=http://localhost:11435 ollama pull qwen3:8b`
- **"file exists" error**: A symlink was created instead of a real directory. Remove it (`rm /local/$USER/.ollama/models`) and let Ollama create the directory itself on first pull.
- **Port already in use**: Another process is using 11435. Pick a different port.

### Process killed or server rebooted mid-run

BLADE can resume from where it left off if the result directory is intact. Just re-run the same command — it will skip completed candidates and continue. If the result directory is corrupted, delete it and start fresh.

### Backslash line continuations break when pasting

Multi-line commands with `\` can fail when pasted into SSH terminals. Always paste commands as a single line.
