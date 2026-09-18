#!/usr/bin/env bash
# Stage 3 (iteration >= 1): retrain on the clean pairs plus the pseudo-labelled
# noisy pairs, starting from the previous iteration.
#
#   MODEL=sd15 INIT_CHECKPOINT=runs/sd15/iter0/checkpoint-1600 ITER=1 bash scripts/3_train_pseudo.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to the model of the previous iteration}"
ITER=${ITER:-1}
DATA_DIR=${DATA_DIR:-data}
PSEUDO_LABELS=${PSEUDO_LABELS:-runs/${MODEL}/iter${ITER}/pseudo_labels/pseudo_labels.json}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}/iter${ITER}}

# DPO reference model: the SD 1.5 runs of the paper used the previous
# iteration's model, the SDXL runs used the base model.
case "${MODEL}" in
  sd15) ACCUM=4; WARMUP=400; REF=(--ref_model_name_or_path "${INIT_CHECKPOINT}"); EXTRA=() ;;
  sdxl) ACCUM=2; WARMUP=50;  REF=(); EXTRA=(--gradient_checkpointing) ;;
esac

launch "${ACCELERATE_CONFIG}" semi_dpo/train.py \
  --model_family "${MODEL}" \
  --train_dataset_name "${DATA_DIR}/pickapic_v2_scored" \
  --pseudo_label_path "${PSEUDO_LABELS}" \
  --pretrained_unet_name_or_path "${INIT_CHECKPOINT}" ${REF[@]+"${REF[@]}"} \
  --val_dataset_name "${DATA_DIR}/consensus/clean" --val_split_name test \
  --output_dir "${OUTPUT_DIR}" \
  --beta_dpo 2500 \
  --train_batch_size 4 --gradient_accumulation_steps "${ACCUM}" \
  --learning_rate 4e-10 --scale_lr \
  --lr_scheduler constant_with_warmup --lr_warmup_steps "${WARMUP}" \
  --max_train_steps 4000 --checkpointing_steps 100 \
  --mixed_precision bf16 --vae_encode_batch_size 16 --dataloader_num_workers 16 \
  --report_to wandb --project_name semi-dpo \
  --seed 0 ${EXTRA[@]+"${EXTRA[@]}"} "$@"
