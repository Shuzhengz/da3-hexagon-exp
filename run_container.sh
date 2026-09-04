#!/usr/bin/env bash
# Runner script for DA3METRIC-LARGE on Qualcomm Rubik Pi 3 NPU Container
set -euo pipefail

IMAGE_NAME="da3-metric-large-npu:latest"
INPUT_PATH="${1:-/home/ubuntu/da3-npu/data/sample.png}"
OUTPUT_DIR="${2:-/home/ubuntu/da3-npu/output}"

mkdir -p "$OUTPUT_DIR"

INPUT_ABS="$(realpath "$INPUT_PATH")"
OUTPUT_ABS="$(realpath "$OUTPUT_DIR")"

echo "================================================================"
echo " Launching DA3METRIC-LARGE NPU Docker Container"
echo " Device: Qualcomm Rubik Pi 3 (Hexagon v68 DSP)"
echo " Image:  $INPUT_ABS"
echo " Output: $OUTPUT_ABS"
echo "================================================================"

docker run --rm \
  --device /dev/fastrpc-cdsp \
  --device /dev/dma_heap \
  -v "$INPUT_ABS":/input.png:ro \
  -v "$OUTPUT_ABS":/app/output \
  "$IMAGE_NAME" /input.png -o /app/output
