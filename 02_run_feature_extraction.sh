#!/bin/bash
set -e

ENV_NAME="speech_env"

source config/datasets.sh

DATA_ROOT="results/preprocessing"
OUTPUT_ROOT="results/features"

CONFIG="config/features/overall_feature_extraction.yaml"

echo "Starting feature extraction pipeline"
echo "Start time: $(date)"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

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

echo "Finished feature extraction"
echo "End time: $(date)"