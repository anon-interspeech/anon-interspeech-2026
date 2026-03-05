#!/bin/bash
set -e

ENV_NAME="speech_env"

CONFIG="config/nested_cross_validation.yaml"
SCRIPT="src/models/nested_cross_val_opt_parallel.py"

EXP_ID="fusion_experiment"

echo "Starting nested cross validation"
echo "Start time: $(date)"
echo "Using config: $CONFIG"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

python "$SCRIPT" \
    --config "$CONFIG" \
    --exp_id "$EXP_ID"

echo "Finished nested cross validation"
echo "End time: $(date)"