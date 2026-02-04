#!/bin/bash
###############################################################################
# Model Conversion Script
#
# Batch converts models to TensorRT format with multiple precisions
#
# Usage: ./scripts/convert_models.sh --input models/pytorch --output models/tensorrt
###############################################################################

set -e

INPUT_DIR="models/pytorch"
OUTPUT_DIR="models/tensorrt"
PRECISIONS=("fp32" "fp16" "int8")

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

mkdir -p ${OUTPUT_DIR}

echo "Converting models from ${INPUT_DIR} to ${OUTPUT_DIR}"

for model in ${INPUT_DIR}/*.pt; do
    model_name=$(basename "$model" .pt)
    
    for precision in "${PRECISIONS[@]}"; do
        echo "Converting ${model_name} to ${precision}..."
        python -m tensorrt.converter \
            --model "${model}" \
            --output "${OUTPUT_DIR}/${model_name}_${precision}.trt" \
            --precision "${precision}" \
            --batch-size 32
    done
done

echo "Conversion complete!"
