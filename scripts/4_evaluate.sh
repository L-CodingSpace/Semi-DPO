#!/usr/bin/env bash
# Generate images for the benchmark prompts and score them.
#   MODEL=sd15 CHECKPOINT=runs/sd15/iter2/checkpoint-4000 NAME=semi_dpo bash scripts/4_evaluate.sh
# Leave CHECKPOINT empty to evaluate the base model, or pass a hub id such as
# mhdang/dpo-sd1.5-text2image-v1 to evaluate a baseline.
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT=${CHECKPOINT:-}
NAME=${NAME:-${MODEL}_base}
EVAL_DIR=${EVAL_DIR:-eval}

BENCHMARKS=(
  "pickapic yuvalkirstain/pickapic_v2 test_unique"
  "hpdv2 zhwang/HPDv2 test"
  "partiprompts nateraw/parti-prompts train"
)

for benchmark in "${BENCHMARKS[@]}"; do
  read -r name dataset split <<< "${benchmark}"
  launch configs/multi_gpu.yaml semi_dpo/evaluate.py \
    --model_family "${MODEL}" ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
    --dataset_name "${dataset}" --dataset_split "${split}" \
    --output_dir "${EVAL_DIR}/${NAME}/${name}" \
    --num_inference_steps 50 --seed 42
done

echo "Compare runs with: python semi_dpo/report.py ${EVAL_DIR}/*/pickapic --baseline ${EVAL_DIR}/${MODEL}_base/pickapic"
