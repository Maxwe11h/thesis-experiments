#!/bin/bash
# Setup script for REL Compute servers (vibranium, geranium, etc.)
# Creates a conda env on /local and installs only the deps needed to run experiments.
#
# Usage:
#   ssh vibranium.liacs.nl
#   # First time: clone then run setup
#   git clone --recurse-submodules https://github.com/Maxwe11h/thesis-experiments.git /local/$USER/thesis
#   cd /local/$USER/thesis && bash setup_server.sh
#
#   # Update existing:
#   cd /local/$USER/thesis && git pull --ff-only && git submodule update --init --recursive && bash setup_server.sh

set -euo pipefail

WORK_DIR="/local/$USER/thesis"
ENV_DIR="/local/$USER/conda_envs/thesis"
PYTHON_VER="3.11"

echo "=== Setting up thesis experiment on $(hostname) ==="
echo "Working directory: $WORK_DIR"
echo "Conda env: $ENV_DIR"

cd "$WORK_DIR"

# --- Create conda env if it doesn't exist ---
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda env (Python $PYTHON_VER) at $ENV_DIR..."
    conda create -y -p "$ENV_DIR" python="$PYTHON_VER"
fi

echo "Activating conda env..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

echo "Python: $(python --version) at $(which python)"

# --- Install dependencies ---
# Install BLADE and LLaMEA as editable, but skip their heavy optional deps
echo ""
echo "=== Installing packages ==="
pip install --no-deps -e ./LLaMEA
pip install --no-deps -e ./BLADE

# Install only the runtime deps we actually need
pip install \
    numpy \
    pandas \
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
    xgboost \
    openai \
    virtualenv

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
python -c "
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
echo "To activate the env:"
echo "  conda activate $ENV_DIR"
echo ""
echo "To run conditions:"
echo "  cd $WORK_DIR"
echo "  nohup python run_conditions.py <conditions...> > logs/run.log 2>&1 &"
echo ""
echo "To monitor:"
echo "  tail -f $WORK_DIR/logs/run.log"
echo "  cat $WORK_DIR/results/<condition>/progress.json | python3 -m json.tool"
