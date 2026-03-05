#!/bin/bash
set -e

ENV_NAME="speech_env"

source config/datasets.sh

DATA_ROOT="data"
OUTPUT_ROOT="results/preprocessing"

CONFIG="config/preprocessing/best_hyperparameters_preprocessing.yaml"

echo "Starting preprocessing pipeline"
echo "Start time: $(date)"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

for DATASET in "${DATASETS[@]}"; do

INPUT_WAVS="${DATA_ROOT}/${DATASET}/wavs"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"

mkdir -p "$OUTPUT_DIR"

echo "Running preprocessing for $DATASET"

python src/data/preprocessing/acoustic_preprocessing_scaled.py \
    --input "$INPUT_WAVS" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --dataset_type "$DATASET"

done

echo "Finished preprocessing"
echo "End time: $(date)"