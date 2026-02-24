#!/bin/bash
# Setup script for REL Compute servers (vibranium, geranium, etc.)
# Run this once per server to clone the repo and install dependencies.
#
# Usage:
#   ssh vibranium.liacs.nl
#   bash setup_server.sh        # if copied over
#   # OR run inline after cloning:
#   cd /local/$USER/thesis && bash setup_server.sh

set -euo pipefail

WORK_DIR="/local/$USER/thesis"
REPO_URL="https://github.com/Maxwe11h/thesis-experiments.git"

echo "=== Setting up thesis experiment on $(hostname) ==="
echo "Working directory: $WORK_DIR"

# --- Clone or update repo ---
if [ -d "$WORK_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$WORK_DIR"
    git pull --ff-only
    git submodule update --init --recursive
else
    echo "Cloning repo..."
    git clone --recurse-submodules "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# --- Python environment via uv ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Init uv project if needed
if [ ! -f "$WORK_DIR/pyproject.toml" ]; then
    uv init
fi

# Install BLADE and LLaMEA as editable + runtime deps
uv add --editable ./BLADE --editable ./LLaMEA
uv add numpy pandas scipy scikit-learn jsonlines ioh configspace ollama

# --- Create results and logs directories ---
mkdir -p "$WORK_DIR/results"
mkdir -p "$WORK_DIR/logs"

# --- Verify Ollama access and model ---
echo ""
echo "=== Checking Ollama ==="
if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | grep -q "qwen3:8b"; then
        echo "qwen3:8b is available"
    else
        echo "Pulling qwen3:8b..."
        ollama pull qwen3:8b
    fi
else
    echo "WARNING: ollama not found. Make sure Ollama is enabled for your account."
    echo "Contact rel@liacs.leidenuniv.nl if not yet activated."
fi

# --- Verify setup ---
echo ""
echo "=== Verifying setup ==="
uv run python -c "
from experiments.run_experiment import CONDITIONS
from experiments.config import BEHAVIORAL_FEATURES, OLLAMA_MODEL
print(f'Model: {OLLAMA_MODEL}')
print(f'Conditions: {len(CONDITIONS)}')
print(f'Features: {len(BEHAVIORAL_FEATURES)}')
print('Setup OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run conditions:"
echo "  cd $WORK_DIR"
echo "  nohup uv run python run_conditions.py <conditions...> > logs/run.log 2>&1 &"
echo ""
echo "To monitor:"
echo "  tail -f $WORK_DIR/logs/run.log"
echo "  cat $WORK_DIR/results/<condition>/progress.json | python3 -m json.tool"
