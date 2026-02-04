#!/bin/bash
###############################################################################
# Benchmark Script for Model Serving Performance
#
# Runs comprehensive performance benchmarks including:
# - Latency tests (P50, P90, P99)
# - Throughput tests (QPS)
# - Model comparison tests
# - Generates reports and visualizations
#
# Usage: ./scripts/benchmark.sh [--duration 60] [--concurrency 10]
###############################################################################

set -e

DURATION=60
CONCURRENCY=10
OUTPUT_DIR="benchmarks/results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

mkdir -p ${OUTPUT_DIR}

echo "Running benchmarks..."
echo "Duration: ${DURATION}s, Concurrency: ${CONCURRENCY}"

# Latency benchmarks
python benchmarks/latency_test.py --duration ${DURATION} --output ${OUTPUT_DIR}/latency.json

# Throughput benchmarks  
python benchmarks/throughput_test.py --duration ${DURATION} --concurrency ${CONCURRENCY} --output ${OUTPUT_DIR}/throughput.json

# Comparison tests
python benchmarks/compare.py --results-dir ${OUTPUT_DIR}

echo "Benchmarks complete! Results in ${OUTPUT_DIR}/"
