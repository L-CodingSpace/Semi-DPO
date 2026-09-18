#!/usr/bin/env bash
# Stage 1 (iteration 0): Diffusion-DPO on the consensus-clean pairs.
#   MODEL=sd15 bash scripts/1_train_clean.sh
#   MODEL=sdxl bash scripts/1_train_clean.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA_DIR=${DATA_DIR:-data}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}/iter0}

# Settings of the paper runs (32 GPUs, global batch 512 for SD 1.5).
# --scale_lr multiplies the learning rate by the global batch size.
case "${MODEL}" in
  sd15) LR=4e-9;  STEPS=1600; ACCUM=4; WARMUP=400; EXTRA=() ;;
  sdxl) LR=2e-10; STEPS=8000; ACCUM=4; WARMUP=50;  EXTRA=(--gradient_checkpointing) ;;
esac

launch "${ACCELERATE_CONFIG}" semi_dpo/train.py \
  --model_family "${MODEL}" \
  --train_dataset_name "${DATA_DIR}/consensus/clean" --train_split_name train \
  --val_dataset_name "${DATA_DIR}/consensus/clean" --val_split_name test \
  --output_dir "${OUTPUT_DIR}" \
  --beta_dpo 2500 \
  --train_batch_size 4 --gradient_accumulation_steps "${ACCUM}" \
  --learning_rate "${LR}" --scale_lr \
  --lr_scheduler constant_with_warmup --lr_warmup_steps "${WARMUP}" \
  --max_train_steps "${STEPS}" --checkpointing_steps 100 \
  --mixed_precision bf16 --vae_encode_batch_size 16 --dataloader_num_workers 16 \
  --report_to wandb --project_name semi-dpo \
  --seed 0 ${EXTRA[@]+"${EXTRA[@]}"} "$@"
