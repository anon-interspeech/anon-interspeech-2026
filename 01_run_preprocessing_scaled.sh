#!/bin/bash

# Example preprocessing pipeline
# Adapt paths to your local environment.

set -e

DATASETS=(
  BostonNamingTest
  MiniMentalStatus
  PhonemicFluency
  RecallWordList
  RecognizeWordList
  VerbalFluency
)

DATA_ROOT="$YOUR_DATA_PATH"
OUTPUT_ROOT="results/preprocessing"

CONFIG="config/preprocessing/best_hyperparameters_preprocessing.yaml"

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