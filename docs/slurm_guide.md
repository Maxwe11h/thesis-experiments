# SLURM Cluster Guide (LIACS REL Compute)

## Cluster overview

| Node | GPUs | VRAM | Partition | Status |
|------|------|------|-----------|--------|
| ceratanium | 4x L40S | 48 GB each | L40s_students | Available |
| saronite | 4x L40S | 48 GB each | L40s_students | Available |
| netherite | 4x L40S | 48 GB each | L40s_students | Available |
| verterium | 4x L40S | 48 GB each | L40s_students | Intermittent |

- Max job duration: **5 days**
- `/local/$USER/` is **per-node** storage (not shared between nodes)
- Home directory (`~/`) is shared across all nodes via NFS

## Connecting

### Step 1: SSH to the LIACS gateway

```bash
ssh s3815129@ssh.liacs.nl
```

You land on a gateway machine (e.g. `molybdenum`). This is just a hop — do not run anything here.

### Step 2: SSH to the SLURM submission node

```bash
ssh fs.cc.liacs.nl
```

You land on `calcium`, the dedicated SLURM submission node.

**Rules for `calcium` (fs.cc.liacs.nl):**
- Do **not** run heavy computations or data transfers here
- Use it only for: submitting jobs (`sbatch`), checking queues (`squeue`), monitoring (`sinfo`)
- Short file edits and git operations are fine

### Step 3: Run commands on compute nodes

For **interactive work** (debugging, setup):

```bash
srun --partition=L40s_students --gres=gpu:1 --time=00:30:00 --nodelist=ceratanium --pty bash
```

For **batch jobs** (experiments):

```bash
sbatch slurm/phase1_ollama.sbatch qwen3.5-4b
```

### Quick-connect one-liner (from local machine)

```bash
ssh -J s3815129@ssh.liacs.nl s3815129@fs.cc.liacs.nl
```

Or add to `~/.ssh/config`:

```
Host liacs-gateway
    HostName ssh.liacs.nl
    User s3815129

Host slurm
    HostName fs.cc.liacs.nl
    User s3815129
    ProxyJump liacs-gateway
```

Then just: `ssh slurm`

## Checking cluster status

```bash
# Node availability
sinfo

# Running/queued jobs
squeue -u $USER

# All jobs on the cluster
squeue

# Detailed node info (GPUs, memory, CPUs)
scontrol show node ceratanium
```

## Storage layout on ceratanium

```
/local/s3815129/
  ollama/                  # Ollama binary (installed)
    bin/ollama
  ollama_models/           # Pulled model weights
  conda_envs/thesis/       # Conda environment (after setup)

~/thesis/                  # Shared repo (accessible from all nodes)
  experiments/
  slurm/
  results_phase1/          # Results written here (shared, persists)
  logs/slurm/              # SLURM job logs
```

**Important:** `/local` is wiped if the node is reprovisioned. Keep results in `~/thesis/results_phase1/` (home directory).

## First-time node setup

Only needs to be done once per node. This installs Ollama, pulls all 8 models, creates the conda env, and installs Python dependencies.

### 1. Clone the repo (if not already on the shared filesystem)

From `calcium`:

```bash
cd ~
git clone --recurse-submodules https://github.com/Maxwe11h/thesis-experiments.git thesis
```

### 2. Run setup on the target node

```bash
srun --partition=L40s_students --gres=gpu:1 --time=01:00:00 --nodelist=ceratanium --pty bash slurm/setup_node.sh
```

This takes ~30 minutes (mostly model downloads: ~70 GB total).

### 3. Verify

```bash
srun --partition=L40s_students --gres=gpu:1 --time=00:10:00 --nodelist=ceratanium bash -c '
    export OLLAMA_MODELS=/local/$USER/ollama_models
    /local/$USER/ollama/bin/ollama serve > /tmp/ollama_test.log 2>&1 &
    sleep 5
    /local/$USER/ollama/bin/ollama run qwen3.5:4b "Say hello in one word"
    kill %1
'
```

## Running Phase 1 experiments

### Submit all models (4 jobs, one per GPU)

```bash
cd ~/thesis
bash slurm/submit_all.sh ceratanium
```

This creates 4 SLURM jobs, each assigned 1 GPU:

| Job | Models | Est. time |
|-----|--------|-----------|
| p1-small-a | qwen3.5-4b, qwen3.5-9b, granite4-3b | ~8h |
| p1-small-b | rnj-1-8b, olmo3-7b | ~6h |
| p1-large-a | devstral-small-2-24b, qwen3.5-27b | ~8h |
| p1-large-b | olmo3-30b | ~5h |

Within each job, all 5 seeds run in parallel.

### Submit a single model

```bash
sbatch --nodelist=ceratanium slurm/phase1_ollama.sbatch qwen3.5-4b
```

### Submit with specific seeds

```bash
sbatch --nodelist=ceratanium slurm/phase1_ollama.sbatch qwen3.5-4b "0 1 2"
```

### Run Gemini models (no GPU / no SLURM needed)

```bash
GOOGLE_API_KEY=your-key python run_phase1.py gemini-3-pro gemini-3-flash
```

Run this from any machine with network access (laptop, vibranium, calcium).

## Monitoring

### Job status

```bash
squeue -u $USER
```

Output columns: JOBID, PARTITION, NAME, USER, ST (status), TIME, NODES, NODELIST

| Status | Meaning |
|--------|---------|
| R | Running |
| PD | Pending (waiting for resources) |
| CG | Completing |

### Live logs

```bash
# Main job log
tail -f logs/slurm/phase1-p1-small-a-<jobid>.out

# Per-seed log
tail -f logs/slurm/phase1-qwen3.5-4b-seed0-<jobid>.log
```

### GPU usage (from within a job)

```bash
srun --jobid=<jobid> --overlap nvidia-smi
```

### Cancel a job

```bash
scancel <jobid>        # cancel one job
scancel -u $USER       # cancel all your jobs
```

### Check completed job info

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Troubleshooting

### "Ollama not found" error in job

Setup hasn't been run on that node. Run `setup_node.sh` on the specific node:

```bash
srun --nodelist=<node> --gres=gpu:1 --time=01:00:00 --pty bash slurm/setup_node.sh
```

### Job stuck in PD (pending)

```bash
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

The REASON column shows why: `Resources` (waiting for GPU), `Priority` (other jobs ahead), `NodeDown`, etc.

### Interactive session shows no prompt

Press Enter a few times. If still nothing after 30 seconds, Ctrl+C and retry with an explicit node:

```bash
srun --partition=L40s_students --gres=gpu:1 --time=00:30:00 --nodelist=ceratanium --pty bash
```

### Port conflict (multiple jobs on same node)

Each job uses a unique Ollama port derived from its SLURM job ID (`11434 + jobid % 100`). Conflicts are unlikely but possible. If Ollama fails to start, check the log:

```bash
cat /tmp/ollama-<jobid>.log
```

### Out of VRAM

Large models (27B, 30B) need ~20 GB. If other processes are using the GPU, the model won't load. Ensure only one model runs per GPU (the scripts handle this).
