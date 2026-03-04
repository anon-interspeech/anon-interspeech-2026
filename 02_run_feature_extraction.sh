#!/bin/bash

# Example feature extraction pipeline

set -e

DATASETS=(
  PhonemicFluency
  BostonNamingTest
  MiniMentalStatus
  RecallWordList
  RecognizeWordList
  VerbalFluency
)

DATA_ROOT="$YOUR_DATA_PATH/processed_audio"
OUTPUT_ROOT="results/features"

CONFIG="config/features/overall_feature_extraction.yaml"

for DATASET in "${DATASETS[@]}"; do

    INPUT_AUDIO="${DATA_ROOT}/${DATASET}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"

    mkdir -p "$OUTPUT_DIR"

    echo "Extracting features for $DATASET"

    python src/features/overall_feature_extraction.py \
        --input "$INPUT_AUDIO" \
        --output "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --dataset_type "$DATASET"

done