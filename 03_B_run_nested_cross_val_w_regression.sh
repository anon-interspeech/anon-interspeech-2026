#!/bin/bash

# Example script for running regression experiments

set -e

CONFIG="config/nested_cross_validation_regression.yaml"
PROCESSING_SCRIPT="src/models/nested_cross_val_opt_parallel_w_regression.py"

EXP_ID="fusion_regression"

echo "Starting regression experiment at $(date)"
echo "Using config: $CONFIG"

python "$PROCESSING_SCRIPT" \
    --config "$CONFIG" \
    --exp_id "$EXP_ID"

echo "Experiment finished at $(date)"