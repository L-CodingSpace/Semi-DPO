#!/usr/bin/env bash
# Stage 0: build the Pick-a-Pic v2 pair dataset, score it with the five reward
# models and split it into clean (all models agree with the human label) and
# noisy pairs.
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA_DIR=${DATA_DIR:-data}

python semi_dpo/prepare_pickapic.py --output_dir "${DATA_DIR}/pickapic_v2"

launch configs/multi_gpu.yaml semi_dpo/score_pairs.py \
  --dataset_name "${DATA_DIR}/pickapic_v2" \
  --work_dir "${DATA_DIR}/pickapic_v2_score_shards" \
  --output_dir "${DATA_DIR}/pickapic_v2_scored"

python semi_dpo/split_consensus.py \
  --dataset_name "${DATA_DIR}/pickapic_v2_scored" \
  --output_dir "${DATA_DIR}/consensus"
