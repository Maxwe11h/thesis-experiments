#!/bin/bash
# Minimal setup for running Gemini API experiments on vibranium.
# No Ollama/vLLM/GPU required — just Python deps + API key.
#
# Usage:
#   ssh vibranium.liacs.nl
#   cd /local/$USER/thesis
#   git pull --ff-only && git submodule update --init --recursive
#   bash setup_vibranium_gemini.sh

set -euo pipefail

WORK_DIR="/local/$USER/thesis"
ENV_DIR="/local/$USER/conda_envs/thesis"
PYTHON_VER="3.11"

echo "=== Gemini-only setup on $(hostname) ==="
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

# --- Install BLADE + LLaMEA ---
echo ""
echo "=== Installing packages ==="
pip install --no-deps -e ./LLaMEA
pip install --no-deps -e ./BLADE

# Runtime deps (same as setup_server.sh, plus google-genai)
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
    pymoo \
    google-genai

# --- Create directories ---
mkdir -p "$WORK_DIR/results_phase1"
mkdir -p "$WORK_DIR/logs"

# --- Check API key ---
echo ""
echo "=== Checking Gemini API key ==="
if [ -f "$WORK_DIR/.env" ]; then
    set -a && source "$WORK_DIR/.env" && set +a
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "WARNING: GOOGLE_API_KEY not set."
    echo "  Option 1: export GOOGLE_API_KEY=your-key"
    echo "  Option 2: Add GOOGLE_API_KEY=your-key to $WORK_DIR/.env"
else
    echo "GOOGLE_API_KEY found: ${GOOGLE_API_KEY:0:8}...${GOOGLE_API_KEY: -4}"
    python -c "
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for model_id in ['gemini-3-pro-preview', 'gemini-3-flash-preview']:
    resp = client.models.generate_content(model=model_id, contents='Reply OK')
    print(f'  {model_id}: OK')
print('API validation passed')
"
fi

# --- Verify BLADE/LLaMEA imports ---
echo ""
echo "=== Verifying imports ==="
python -c "
from experiments.phase1_config import CANDIDATE_MODELS
gemini = {k: v for k, v in CANDIDATE_MODELS.items() if v['type'] == 'gemini'}
print(f'Gemini models registered: {len(gemini)}')
for tag, cfg in gemini.items():
    print(f'  {tag}: {cfg[\"model\"]}')
print('Import verification passed')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run Gemini experiments:"
echo "  conda activate $ENV_DIR"
echo "  cd $WORK_DIR"
echo "  set -a && source .env && set +a"
echo ""
echo "  # Run both models (all 5 seeds):"
echo "  nohup python run_phase1.py gemini-3-flash > logs/gemini-flash.log 2>&1 &"
echo "  nohup python run_phase1.py gemini-3-pro > logs/gemini-pro.log 2>&1 &"
echo ""
echo "  # Or run all 10 seeds in parallel:"
echo "  for model in gemini-3-flash gemini-3-pro; do"
echo "    for seed in 0 1 2 3 4; do"
echo "      nohup python run_phase1.py \$model --seeds \$seed > logs/\${model}_s\${seed}.log 2>&1 &"
echo "    done"
echo "  done"
echo ""
echo "  # Monitor:"
echo "  tail -f logs/gemini-flash.log"
