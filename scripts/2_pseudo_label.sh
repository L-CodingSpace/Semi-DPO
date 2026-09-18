#!/usr/bin/env bash
# Stage 2: use the model of the previous iteration as an implicit classifier.
#   1. Per-timestep accuracy on the held-out clean pairs (Table 7 of the paper).
#   2. Per-timestep confidence on every training pair.
#   3. Select high-confidence pseudo-labels per timestep interval.
#
#   MODEL=sd15 CHECKPOINT=runs/sd15/iter0/checkpoint-1600 ITER=1 bash scripts/2_pseudo_label.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${CHECKPOINT:?Set CHECKPOINT to the model of the previous iteration}"
ITER=${ITER:-1}
DATA_DIR=${DATA_DIR:-data}
WORK_DIR=${WORK_DIR:-runs/${MODEL}/iter${ITER}/pseudo_labels}
PAST_LABELS=${PAST_LABELS:-}  # pseudo-label files of earlier iterations, space separated
# Clean pairs that anchor this iteration with their human label.  The SD 1.5
# paper runs used a different third of the clean pairs in every iteration
# (see README, "Training details").
CLEAN_IDX=${CLEAN_IDX:-${DATA_DIR}/consensus/clean_idx.json}

# Paper: 80th percentile per interval, raised where the accuracy on the clean
# test pairs drops below 70% (t > 650 for SD 1.5).  Inspect
# ${WORK_DIR}/test/summary.csv and adjust before building the labels.
SELECT_MODE=${SELECT_MODE:-percentile}
SELECT_VALUES=${SELECT_VALUES:-"80 80 80 80 80 80 90 90 90 90"}

case "${MODEL}" in
  sd15) BATCH=32 ;;
  sdxl) BATCH=16 ;;
esac

for split in test train; do
  if [ "${split}" = test ]; then
    dataset=(--dataset_name "${DATA_DIR}/consensus/clean" --dataset_split test)
  else
    dataset=(--dataset_name "${DATA_DIR}/pickapic_v2_scored")
  fi
  launch configs/multi_gpu.yaml semi_dpo/compute_confidence.py \
    --model_family "${MODEL}" --checkpoint "${CHECKPOINT}" "${dataset[@]}" \
    --output_dir "${WORK_DIR}/${split}" \
    --beta_dpo 2500 --batch_size "${BATCH}" --mixed_precision bf16 --dataloader_num_workers 16
done

python semi_dpo/summarize_confidence.py --confidence_dir "${WORK_DIR}/test"

# shellcheck disable=SC2086
python semi_dpo/build_pseudo_labels.py \
  --confidence_dir "${WORK_DIR}/train" \
  --clean_idx_path "${CLEAN_IDX}" \
  --mode "${SELECT_MODE}" --values ${SELECT_VALUES} \
  ${PAST_LABELS:+--past_pseudo_labels ${PAST_LABELS}} \
  --output "${WORK_DIR}/pseudo_labels.json"
