#!/bin/bash
set -e

ENV_NAME="speech_env"
SCRIPT="src/utils/audio_qc_pipeline.py"

DATA_ROOT="data"
OUTPUT_ROOT="results/qc"

source config/datasets.sh

echo "Starting optional audio quality control"
echo "Start time: $(date)"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

for DATASET in "${DATASETS[@]}"; do

INPUT_DIR="${DATA_ROOT}/${DATASET}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"

mkdir -p "$OUTPUT_DIR"

echo "Running QC for $DATASET"

python "$SCRIPT" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --dataset_type "$DATASET"

done

echo "Finished audio QC"
echo "End time: $(date)"