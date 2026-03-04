#!/bin/bash

set -e

CONFIG="config/mci_winner.yaml"
SCRIPT="src/final_training_best_model/final_train.py"

echo "Starting holdout evaluation..."
echo "Using config: $CONFIG"
echo "Running script: $SCRIPT"
echo "Start time: $(date)"

source ~/.bashrc

if command -v conda &> /dev/null; then
    conda activate speech_ssh
fi

python -u $SCRIPT

echo "Finished holdout evaluation"
echo "End time: $(date)"