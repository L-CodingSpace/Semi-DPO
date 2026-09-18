#!/usr/bin/env python
"""Score every preference pair with a trained model at each timestep anchor.

For a pair (winner, loser) and anchor ``t`` the confidence is the implicit
Diffusion-DPO logit

    z_t = beta * [(ref_w - ref_l) - (model_w - model_l)]

where ``*_w`` / ``*_l`` are denoising losses of the winner and loser at ``t``.
``z_t > 0`` means the model agrees with the human label.  Winner and loser
share the same noise, and one noise draw is reused across all anchors.

Each process writes ``confidence_rank{r}.jsonl`` into ``--output_dir``; one
line per pair: ``{"idx": 12, "timestep_50": 1.7, ..., "timestep_950": -0.3}``.
Run on the held-out clean split to measure per-timestep accuracy
(``summarize_confidence.py``) and on the full training set to generate
pseudo-labels (``build_pseudo_labels.py``).
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from semi_dpo import data as data_lib
from semi_dpo.losses import per_sample_mse
from semi_dpo.models import (
    FAMILIES,
    TextConditioner,
    encode_images,
    load_noise_scheduler,
    load_unet,
    load_vae,
    noise_target,
)
from semi_dpo.pseudo_labels import TIMESTEP_ANCHORS, timestep_key
from semi_dpo.utils import setup_logging, weight_dtype_for

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the model acting as the classifier.")
    parser.add_argument("--pretrained_model_name_or_path", default=None, help="Base model; defaults per family.")
    parser.add_argument("--ref_model_name_or_path", default=None, help="Reference UNet; defaults to the base model.")
    parser.add_argument("--pretrained_vae_model_name_or_path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta_dpo", type=float, default=2500)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vae_encode_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    family = FAMILIES[args.model_family]
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or family.default_model
    args.ref_model_name_or_path = args.ref_model_name_or_path or args.pretrained_model_name_or_path
    args.resolution = args.resolution or family.default_resolution
    return args


@torch.no_grad()
def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    setup_logging()
    set_seed(args.seed, device_specific=True)
    family = FAMILIES[args.model_family]
    dtype = weight_dtype_for(accelerator.mixed_precision)
    device = accelerator.device
    base = args.pretrained_model_name_or_path

    scheduler = load_noise_scheduler(base)
    conditioner = TextConditioner(args.model_family, base, args.revision, args.variant).to(device, dtype)
    vae = load_vae(base, args.pretrained_vae_model_name_or_path, args.revision, args.variant)
    vae.to(device, dtype=torch.float32 if family.vae_in_fp32 else dtype)

    unet = load_unet(base, args.checkpoint, args.revision, args.variant).to(device, dtype).eval()
    ref_checkpoint = None if args.ref_model_name_or_path == base else args.ref_model_name_or_path
    ref_unet = load_unet(base, ref_checkpoint, args.revision, args.variant).to(device, dtype).eval()

    dataset = data_lib.load_split(args.dataset_name, args.dataset_split)
    if "idx" not in dataset.column_names:
        raise ValueError("The dataset needs an integer `idx` column to key the confidences")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    # Deterministic preprocessing: no random crop or flip.
    dataset = dataset.with_transform(
        partial(
            data_lib.preprocess_pairs,
            transform=data_lib.image_transforms(args.resolution),
            tokenize=conditioner.tokenize,
        )
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_lib.collate_pairs,
        num_workers=args.dataloader_num_workers,
    )
    dataloader = accelerator.prepare(dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_path = os.path.join(args.output_dir, f"confidence_rank{accelerator.process_index:03d}.jsonl")
    logger.info(f"{len(dataset)} pairs, {len(dataloader)} batches per process, writing to {args.output_dir}")

    with open(shard_path, "w", encoding="utf-8") as shard:
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            latents = encode_images(vae, torch.cat(pixel_values.chunk(2, dim=1)), args.vae_encode_batch_size)
            latents = latents.to(dtype)
            num_pairs = latents.shape[0] // 2
            noise = torch.randn_like(latents[:num_pairs]).repeat(2, 1, 1, 1)
            cond = conditioner.encode(batch, args.resolution, dtype).repeat(2)

            rows = [{"idx": int(idx)} for idx in batch["idx"].tolist()]
            for anchor in TIMESTEP_ANCHORS:
                timesteps = torch.full((2 * num_pairs,), anchor, device=device, dtype=torch.long)
                noisy = scheduler.add_noise(latents, noise, timesteps)
                target = noise_target(scheduler, latents, noise, timesteps)

                model_w, model_l = per_sample_mse(unet(noisy, timesteps, **cond.unet_kwargs()).sample, target).chunk(2)
                ref_w, ref_l = per_sample_mse(ref_unet(noisy, timesteps, **cond.unet_kwargs()).sample, target).chunk(2)
                confidence = args.beta_dpo * ((ref_w - ref_l) - (model_w - model_l))

                for row, value in zip(rows, confidence.tolist()):
                    row[timestep_key(anchor)] = value

            for row in rows:
                shard.write(json.dumps(row) + "\n")
            shard.flush()

    accelerator.wait_for_everyone()
    logger.info("Done")


if __name__ == "__main__":
    main(parse_args())
