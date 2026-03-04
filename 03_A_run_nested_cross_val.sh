#!/bin/bash

# Example script for running nested cross-validation experiments

set -e

CONFIG="config/nested_cross_validation.yaml"
PROCESSING_SCRIPT="src/models/nested_cross_val_opt_parallel.py"

EXP_ID="fusion_experiment"

echo "Starting experiment at $(date)"
echo "Using config: $CONFIG"

python "$PROCESSING_SCRIPT" \
    --config "$CONFIG" \
    --exp_id "$EXP_ID"

echo "Experiment finished at $(date)"