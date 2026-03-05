#!/bin/bash
set -e

ENV_NAME="speech_env"

CONFIG="config/mci_winner.yaml"
SCRIPT="src/models/run_bakery.py"

echo "Starting holdout validation"
echo "Start time: $(date)"
echo "Using config: $CONFIG"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

python -u "$SCRIPT" \
    --config "$CONFIG"

echo "Finished holdout validation"
echo "End time: $(date)"