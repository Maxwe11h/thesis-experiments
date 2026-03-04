#!/bin/bash
# One-time setup for a SLURM L40S node.
# Run via: srun --partition=L40s_students --gres=gpu:1 --time=01:00:00 --nodelist=ceratanium --pty bash slurm/setup_node.sh
#
# What it does:
#   1. Installs Ollama to /local/$USER/ollama/ (if not already present)
#   2. Pulls all 8 Ollama models
#   3. Creates conda env at /local/$USER/conda_envs/thesis
#   4. Installs Python dependencies
#
# NOTE: /local is per-node, so run this on each node you want to use.

set -euo pipefail

NODE=$(hostname)
OLLAMA_DIR="/local/$USER/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
export OLLAMA_MODELS="/local/$USER/ollama_models"
CONDA_ENV="/local/$USER/conda_envs/thesis"
REPO_DIR="$HOME/thesis"
PYTHON_VER="3.11"

echo "=== Phase 1 SLURM node setup on $NODE ==="
echo "  Ollama:    $OLLAMA_DIR"
echo "  Models:    $OLLAMA_MODELS"
echo "  Conda env: $CONDA_ENV"
echo "  Repo:      $REPO_DIR"
echo ""

# ---- 1. Install Ollama ----
if [ -x "$OLLAMA_BIN" ]; then
    echo "[1/4] Ollama already installed: $($OLLAMA_BIN --version)"
else
    echo "[1/4] Installing Ollama..."
    mkdir -p "$OLLAMA_DIR"
    curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C "$OLLAMA_DIR"
    echo "  Installed: $($OLLAMA_BIN --version)"
fi

# ---- 2. Pull models ----
echo ""
echo "[2/4] Pulling Ollama models (starting server temporarily)..."
mkdir -p "$OLLAMA_MODELS"
$OLLAMA_BIN serve > /tmp/ollama_setup.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null; wait $OLLAMA_PID 2>/dev/null" EXIT

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/ > /dev/null 2>&1; then
        echo "  Server ready"
        break
    fi
    sleep 1
done

MODELS=(
    "qwen3.5:4b"
    "qwen3.5:9b"
    "qwen3.5:27b"
    "rnj-1:8b"
    "devstral-small-2:24b"
    "olmo-3:7b"
    "olmo-3:32b"
    "granite4:3b"
)

for model in "${MODELS[@]}"; do
    echo "  Pulling $model..."
    $OLLAMA_BIN pull "$model"
done

echo "  All models pulled. Available:"
$OLLAMA_BIN list

# Stop Ollama server
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null || true
trap - EXIT

# ---- 3. Create conda env ----
echo ""
echo "[3/4] Setting up conda environment..."
eval "$(conda shell.bash hook)"

if [ -d "$CONDA_ENV" ]; then
    echo "  Conda env already exists at $CONDA_ENV"
else
    echo "  Creating conda env (Python $PYTHON_VER)..."
    conda create -y -p "$CONDA_ENV" python="$PYTHON_VER"
fi

conda activate "$CONDA_ENV"
echo "  Python: $(python --version) at $(which python)"

# ---- 4. Install dependencies ----
echo ""
echo "[4/4] Installing Python packages..."
cd "$REPO_DIR"

pip install --no-deps -e ./LLaMEA
pip install --no-deps -e ./BLADE

pip install \
    cloudpickle \
    numpy \
    "pandas>=2.2.3,<3" \
    scipy \
    scikit-learn \
    ioh \
    ollama \
    jsonlines \
    configspace \
    joblib \
    tqdm \
    lizard \
    networkx \
    "xgboost>=2.1.1,<3" \
    "openai>=1.99.1,<2" \
    virtualenv \
    antropy \
    nolds \
    pymoo

# ---- Done ----
echo ""
echo "=== Setup complete on $NODE ==="
echo ""
echo "Available models:"
echo "  ${MODELS[*]}"
echo ""
echo "To verify:"
echo "  srun --partition=L40s_students --nodelist=$NODE --gres=gpu:1 --time=00:10:00 bash -c '"
echo "    export OLLAMA_MODELS=/local/\$USER/ollama_models"
echo "    /local/\$USER/ollama/bin/ollama serve &"
echo "    sleep 3 && /local/\$USER/ollama/bin/ollama run qwen3.5:4b \"Hello\" && kill %1'"
