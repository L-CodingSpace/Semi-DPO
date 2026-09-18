#!/usr/bin/env python
# Copyright 2024 bram-w, The HuggingFace Inc. team, and the Semi-DPO authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion-DPO training for SD 1.5 and SDXL, with optional pseudo-labels.

* Stage 1 (clean data): plain Diffusion-DPO on the consensus-filtered set.
* Stage 3 (self-training): pass ``--pseudo_label_path``.  Each pair then gets
  a timestep drawn from its labelled intervals and the preference direction of
  that interval (``+1`` / ``-1``); unlabelled pairs are masked out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import convert_state_dict_to_diffusers
from tqdm.auto import tqdm

from semi_dpo import data as data_lib
from semi_dpo import pseudo_labels as pl
from semi_dpo.losses import LOSS_TYPES, dpo_loss, per_sample_mse, reward_consensus_weight
from semi_dpo.models import (
    FAMILIES,
    LORA_WEIGHT_NAME,
    TextConditioner,
    build_pipeline,
    encode_images,
    enforce_zero_terminal_snr,
    load_noise_scheduler,
    load_unet,
    load_vae,
    noise_target,
    use_dpm_solver,
)
from semi_dpo.utils import adapters_disabled, clear_cache, setup_logging, weight_dtype_for

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("model")
    group.add_argument("--model_family", choices=sorted(FAMILIES), required=True)
    group.add_argument("--pretrained_model_name_or_path", default=None, help="Base model; defaults per family.")
    group.add_argument(
        "--pretrained_unet_name_or_path",
        default=None,
        help="Checkpoint to initialise the trained UNet from (e.g. the stage-1 model).",
    )
    group.add_argument(
        "--ref_model_name_or_path",
        default=None,
        help="Frozen DPO reference UNet. Defaults to the base model.",
    )
    group.add_argument("--pretrained_vae_model_name_or_path", default=None)
    group.add_argument("--revision", default=None)
    group.add_argument("--variant", default=None)
    group.add_argument("--is_turbo", action="store_true", help="Fine-tune SDXL-Turbo on its 4-step schedule.")

    group = parser.add_argument_group("data")
    group.add_argument("--train_dataset_name", required=True)
    group.add_argument("--train_split_name", default=None)
    group.add_argument("--pseudo_label_path", default=None, help="Pseudo-label json from build_pseudo_labels.py.")
    group.add_argument("--val_dataset_name", default=None)
    group.add_argument("--val_split_name", default=None)
    group.add_argument("--num_val_samples", type=int, default=50)
    group.add_argument("--max_train_samples", type=int, default=None)
    group.add_argument("--filter_bad_samples", default=None, help="Text file with one row position per line to drop.")
    group.add_argument("--resolution", type=int, default=None, help="Defaults to 512 (sd15) / 1024 (sdxl).")
    group.add_argument("--random_crop", action="store_true")
    group.add_argument("--no_hflip", action="store_true", help="Disable random flips (SD 1.5 only; SDXL never flips).")
    group.add_argument("--random_swap_label_prob", type=float, default=0.0, help="Label-noise ablation.")
    group.add_argument("--dataloader_num_workers", type=int, default=0)
    group.add_argument("--cache_dir", default=None)

    group = parser.add_argument_group("dpo")
    group.add_argument("--beta_dpo", type=float, default=2500)
    group.add_argument("--loss_type", choices=LOSS_TYPES, default="sigmoid")
    group.add_argument("--gce_q", type=float, default=0.5)
    group.add_argument("--sce_beta", type=float, default=0.5)
    group.add_argument("--log_zero", type=float, default=-6.0, help="log(0) surrogate of the SCE reverse term.")
    group.add_argument("--robust_loss_start_step", type=int, default=-1)
    group.add_argument("--robust_loss_end_step", type=int, default=2**62)
    group.add_argument(
        "--use_reward_scores",
        action="store_true",
        help="Weight the gce loss by the fraction of reward models disagreeing with the human label.",
    )
    group.add_argument("--truncate_timesteps", type=int, default=-1, help="Zero the robust weight for t >= this.")

    group = parser.add_argument_group("optimisation")
    group.add_argument("--train_batch_size", type=int, default=4, help="Per device.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    group.add_argument("--max_train_steps", type=int, default=None)
    group.add_argument("--num_train_epochs", type=int, default=1)
    group.add_argument("--learning_rate", type=float, default=1e-8)
    group.add_argument(
        "--scale_lr",
        action="store_true",
        help="Multiply the learning rate by the total batch size (per-device batch x accumulation x GPUs).",
    )
    group.add_argument("--lr_scheduler", default="constant_with_warmup")
    group.add_argument("--lr_warmup_steps", type=int, default=0)
    group.add_argument("--lr_num_cycles", type=int, default=1)
    group.add_argument("--lr_power", type=float, default=1.0)
    group.add_argument("--adam_beta1", type=float, default=0.9)
    group.add_argument("--adam_beta2", type=float, default=0.999)
    group.add_argument("--adam_weight_decay", type=float, default=1e-2)
    group.add_argument("--adam_epsilon", type=float, default=1e-8)
    group.add_argument("--use_8bit_adam", action="store_true")
    group.add_argument("--max_grad_norm", type=float, default=1.0)
    group.add_argument("--gradient_checkpointing", action="store_true")
    group.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    group.add_argument("--allow_tf32", action="store_true")
    group.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    group.add_argument("--vae_encode_batch_size", type=int, default=8)
    group.add_argument("--seed", type=int, default=None)

    group = parser.add_argument_group("lora / ema / reference updates")
    group.add_argument("--use_lora", action="store_true")
    group.add_argument("--lora_rank", type=int, default=None)
    group.add_argument("--use_ema", action="store_true")
    group.add_argument("--ema_decay", type=float, default=0.9999)
    group.add_argument(
        "--update_ref_model_steps", type=int, default=None, help="Copy the policy into the reference every N steps."
    )
    group.add_argument("--update_ref_model_per_epoch", action="store_true")

    group = parser.add_argument_group("checkpointing and logging")
    group.add_argument("--output_dir", required=True)
    group.add_argument("--checkpointing_steps", type=int, default=500)
    group.add_argument("--save_training_state", action="store_true", help="Also save optimizer state for resuming.")
    group.add_argument("--resume_from_checkpoint", default=None, help='A checkpoint directory or "latest".')
    group.add_argument("--run_validation", action="store_true")
    group.add_argument("--validation_steps", type=int, default=200)
    group.add_argument("--clear_cache_steps", type=int, default=100)
    group.add_argument("--report_to", default="tensorboard")
    group.add_argument("--project_name", default="semi-dpo")
    group.add_argument("--logging_dir", default="logs")
    group.add_argument("--wandb_resume_id", default=None)
    group.add_argument("--push_to_hub", action="store_true")
    group.add_argument("--hub_model_id", default=None)
    group.add_argument("--hub_token", default=None)

    args = parser.parse_args(argv)
    family = FAMILIES[args.model_family]
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or family.default_model
    args.ref_model_name_or_path = args.ref_model_name_or_path or args.pretrained_model_name_or_path
    args.resolution = args.resolution or family.default_resolution

    if args.use_lora and args.lora_rank is None:
        parser.error("--use_lora requires --lora_rank")
    if not args.use_lora and args.lora_rank is not None:
        parser.error("--lora_rank is only used with --use_lora")
    if args.use_lora and (args.update_ref_model_steps or args.update_ref_model_per_epoch):
        parser.error("Reference model updates are only supported for full fine-tuning")
    if args.is_turbo and "turbo" not in args.pretrained_model_name_or_path:
        parser.error("--is_turbo expects an SDXL-Turbo base model")
    if args.is_turbo and args.pseudo_label_path:
        parser.error("--is_turbo cannot be combined with timestep pseudo-labels")
    if args.pseudo_label_path and args.random_swap_label_prob > 0:
        parser.error("--random_swap_label_prob would corrupt the pseudo-labels")
    if args.update_ref_model_steps is not None and args.update_ref_model_steps <= 0:
        parser.error("--update_ref_model_steps must be positive")
    if args.report_to == "wandb" and args.hub_token is not None:
        parser.error("Do not combine --report_to=wandb with --hub_token; use `huggingface-cli login` instead")
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_datasets(args, accelerator, conditioner):
    train_dataset = data_lib.load_split(args.train_dataset_name, args.train_split_name, args.cache_dir)

    # Row positions refer to the full dataset, so filter before selecting pseudo-labelled pairs.
    if args.filter_bad_samples:
        with open(args.filter_bad_samples, "r", encoding="utf-8") as handle:
            bad = {int(line) for line in handle if line.strip()}
        keep = [i for i in range(len(train_dataset)) if i not in bad]
        logger.info(f"Dropping {len(train_dataset) - len(keep)} rows listed in {args.filter_bad_samples}")
        train_dataset = train_dataset.select(keep)

    pseudo_labels = None
    if args.pseudo_label_path:
        pseudo_labels = pl.load_pseudo_labels(args.pseudo_label_path)
        train_dataset = data_lib.select_indices(train_dataset, pseudo_labels.keys())
        logger.info(f"Loaded {len(pseudo_labels)} pseudo-labelled pairs, {len(train_dataset)} found in the dataset")
        if len(train_dataset) == 0:
            raise ValueError(
                f"None of the {len(pseudo_labels)} pairs in {args.pseudo_label_path} is in {args.train_dataset_name}; "
                "the pseudo-labels must be keyed by the `idx` column of the training dataset"
            )
        for key, counts in pl.label_histogram(pseudo_labels).items():
            logger.info(f"  {key}: +1={counts[1]} -1={counts[-1]} 0={counts[0]}")

    if args.val_dataset_name:
        val_dataset = data_lib.load_split(args.val_dataset_name, args.val_split_name, args.cache_dir)
    else:
        # A fixed seed keeps the split identical on every rank.
        split = train_dataset.train_test_split(
            test_size=args.num_val_samples, seed=args.seed if args.seed is not None else 0
        )
        train_dataset, val_dataset = split["train"], split["test"]
    validation_prompts = list(val_dataset[data_lib.prompt_column(val_dataset)])[: args.num_val_samples]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    family = FAMILIES[args.model_family]
    transform = data_lib.image_transforms(
        args.resolution, random_crop=args.random_crop, hflip=family.supports_hflip and not args.no_hflip
    )
    train_dataset = train_dataset.with_transform(
        partial(
            data_lib.preprocess_pairs,
            transform=transform,
            tokenize=conditioner.tokenize,
            pseudo_labels=pseudo_labels,
            random_swap_prob=args.random_swap_label_prob,
        )
    )
    loader_kwargs = {}
    if args.dataloader_num_workers > 0:
        loader_kwargs = {"prefetch_factor": 2, "persistent_workers": True}
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_lib.collate_pairs,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **loader_kwargs,
    )
    return train_dataset, train_dataloader, validation_prompts


def sample_timesteps(args, batch, num_pairs, scheduler, device):
    """Timesteps (repeated for winner and loser) and per-pair labels."""
    num_train_timesteps = scheduler.config.num_train_timesteps
    if "pseudo_labels" in batch:
        timesteps, labels = pl.sample_timesteps_for_batch(batch["pseudo_labels"], max_timestep=num_train_timesteps)
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
        labels = torch.tensor(labels, device=device, dtype=torch.float)
    else:
        timesteps = torch.randint(0, num_train_timesteps, (num_pairs,), device=device, dtype=torch.long)
        if args.is_turbo:
            timesteps = 250 * (timesteps % 4) + 249
        labels = None
    return timesteps.repeat(2), labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class Validator:
    """Generates images for fixed prompts and scores them with the reward models."""

    def __init__(self, args, accelerator, vae, weight_dtype, prompts):
        from semi_dpo.reward_models import RewardScorer

        self.args = args
        self.accelerator = accelerator
        self.vae = vae
        self.weight_dtype = weight_dtype
        self.prompts = prompts
        self.scorer = RewardScorer(device=accelerator.device)
        self.baseline = None

    def _unet_for(self, unet, checkpoint_dir):
        if self.accelerator.distributed_type == DistributedType.FSDP:
            # Sharded parameters cannot run a pipeline; reload the saved weights.
            source = checkpoint_dir or self.args.pretrained_unet_name_or_path
            return load_unet(self.args.pretrained_model_name_or_path, source, self.args.revision, self.args.variant)
        return self.accelerator.unwrap_model(unet)

    @torch.no_grad()
    def run(self, unet, step, checkpoint_dir=None):
        args, accelerator = self.args, self.accelerator
        family = FAMILIES[args.model_family]
        logger.info(f"Running validation on {len(self.prompts)} prompts at step {step}")

        pipeline = build_pipeline(
            args.model_family,
            args.pretrained_model_name_or_path,
            unet=self._unet_for(unet, checkpoint_dir),
            vae=self.vae,
            dtype=self.weight_dtype,
            revision=args.revision,
            variant=args.variant,
        ).to(accelerator.device)
        if args.model_family == "sd15":
            use_dpm_solver(pipeline)

        call_kwargs = {"num_inference_steps": family.validation_steps}
        if family.validation_guidance is not None:
            call_kwargs["guidance_scale"] = family.validation_guidance
        if args.is_turbo:
            call_kwargs = {"num_inference_steps": 4, "guidance_scale": 0.0}
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        images, scores = [], {}
        for prompt in self.prompts:
            with torch.autocast(
                accelerator.device.type, dtype=self.weight_dtype, enabled=self.weight_dtype != torch.float32
            ):
                image = pipeline(prompt, generator=generator, **call_kwargs).images[0]
            images.append(image)
            for name, values in self.scorer.score(prompt, image).items():
                scores.setdefault(name, []).extend(values)
        del pipeline

        if self.baseline is None:
            self.baseline = scores
        win_rates = {
            name: 100.0 * np.mean([cur > base for cur, base in zip(values, self.baseline[name])])
            for name, values in scores.items()
        }

        metrics = {f"validation/{name}": float(np.mean(values)) for name, values in scores.items()}
        if step > 0:
            metrics.update({f"validation/{name}_win_rate": rate for name, rate in win_rates.items()})
        accelerator.log(metrics, step=step)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_images(
                    "validation", np.stack([np.asarray(img) for img in images]), step, dataformats="NHWC"
                )
            elif tracker.name == "wandb":
                import wandb

                tracker.log(
                    {
                        "validation": [
                            wandb.Image(img, caption=f"{i}: {p}")
                            for i, (img, p) in enumerate(zip(images, self.prompts))
                        ]
                    },
                    step=step,
                )

        if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
            with open(os.path.join(checkpoint_dir, "validation_scores.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {"step": step, "prompts": self.prompts, "scores": scores, "win_rates": win_rates}, handle, indent=2
                )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def lora_pipeline_class(model_family):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    return StableDiffusionXLPipeline if model_family == "sdxl" else StableDiffusionPipeline


def save_unet(args, accelerator, unet, save_dir):
    """Write the trained weights to ``save_dir/unet`` (all ranks must call this)."""
    if args.use_lora:
        if accelerator.is_main_process:
            from peft.utils import get_peft_model_state_dict

            layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
            lora_pipeline_class(args.model_family).save_lora_weights(
                save_dir, unet_lora_layers=layers, weight_name=LORA_WEIGHT_NAME
            )
        return

    os.makedirs(os.path.join(save_dir, "unet"), exist_ok=True)
    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(unet, StateDictType.FULL_STATE_DICT, config):
            state_dict = unet.state_dict()
        accelerator.save(state_dict, os.path.join(save_dir, "unet", "unet.pt"))
    elif accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_dir, "unet"))


def copy_policy_into_reference(accelerator, unet, ref_unet):
    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(unet, writeback=False), FSDP.summon_full_params(ref_unet, writeback=True):
            source = dict(accelerator.unwrap_model(unet).named_parameters())
            for name, param in accelerator.unwrap_model(ref_unet).named_parameters():
                param.data.copy_(source[name].data)
    else:
        ref_unet.load_state_dict(accelerator.unwrap_model(unet).state_dict())
    ref_unet.eval()


def resolve_resume_path(args):
    if args.resume_from_checkpoint != "latest":
        return args.resume_from_checkpoint
    if not os.path.isdir(args.output_dir):
        return None
    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return os.path.join(args.output_dir, max(checkpoints, key=lambda d: int(d.split("-")[1])))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main(args):
    family = FAMILIES[args.model_family]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=str(Path(args.output_dir, args.logging_dir))
        ),
    )
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    if args.use_ema and is_fsdp:
        raise ValueError("--use_ema is not supported with FSDP")
    if args.use_lora and is_fsdp:
        raise ValueError("--use_lora is not supported with FSDP; use configs/multi_gpu.yaml")
    if is_fsdp and args.run_validation and args.validation_steps % args.checkpointing_steps:
        raise ValueError("With FSDP, --validation_steps must be a multiple of --checkpointing_steps")
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    setup_logging()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = weight_dtype_for(accelerator.mixed_precision)

    # Models --------------------------------------------------------------
    base = args.pretrained_model_name_or_path
    noise_scheduler = load_noise_scheduler(base)
    if args.is_turbo:
        enforce_zero_terminal_snr(noise_scheduler)

    conditioner = TextConditioner(args.model_family, base, args.revision, args.variant).to(
        accelerator.device, weight_dtype
    )
    vae = load_vae(base, args.pretrained_vae_model_name_or_path, args.revision, args.variant)
    vae_dtype = torch.float32 if family.vae_in_fp32 else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)

    unet = load_unet(base, args.pretrained_unet_name_or_path, args.revision, args.variant)
    ref_unet = None
    if args.use_lora:
        from peft import LoraConfig

        unet.requires_grad_(False)
        unet.add_adapter(
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
    else:
        # The reference is loaded like the policy: from the base model, a hub
        # id, or a training checkpoint.
        ref_checkpoint = None if args.ref_model_name_or_path == base else args.ref_model_name_or_path
        ref_unet = load_unet(base, ref_checkpoint, args.revision, args.variant)
        ref_unet.requires_grad_(False)
        ref_unet.to(accelerator.device, dtype=weight_dtype).eval()
    unet.train()

    # Only trainable parameters are kept in fp32 under mixed precision.
    if accelerator.mixed_precision != "no":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if ref_unet is not None:
            ref_unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    ema_unet = None
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), decay=args.ema_decay, model_cls=type(unet), model_config=unet.config)
        ema_unet.to(accelerator.device)

    def save_model_hook(models, weights, output_dir):
        if ema_unet is not None and accelerator.is_main_process:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
        if args.use_lora:
            # LoRA weights are written by save_unet; skip the full model dump.
            while weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        if ema_unet is not None:
            loaded = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), type(unet))
            ema_unet.load_state_dict(loaded.state_dict())
            ema_unet.to(accelerator.device)
        if args.use_lora:
            from diffusers.utils import convert_unet_state_dict_to_peft
            from peft import set_peft_model_state_dict

            lora_state_dict, _ = lora_pipeline_class(args.model_family).lora_state_dict(
                input_dir, weight_name=LORA_WEIGHT_NAME
            )
            unet_state = {k.replace("unet.", "", 1): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state = convert_unet_state_dict_to_peft(unet_state)
            while models:
                result = set_peft_model_state_dict(
                    accelerator.unwrap_model(models.pop()), unet_state, adapter_name="default"
                )
                missing = [k for k in result.missing_keys if "lora" in k]
                if result.unexpected_keys or missing:
                    raise RuntimeError(
                        f"LoRA weights in {input_dir} do not match the model: "
                        f"unexpected {result.unexpected_keys[:3]}, missing {missing[:3]}"
                    )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Data and optimisation -----------------------------------------------
    train_dataset, train_dataloader, validation_prompts = build_datasets(args, accelerator, conditioner)

    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_steps_from_epochs = args.max_train_steps is None
    if max_steps_from_epochs:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if ref_unet is not None and is_fsdp:
        # FSDP needs every model prepared with an optimizer; this one is never stepped.
        dummy_optimizer = torch.optim.AdamW(ref_unet.parameters(), lr=0.0)
        ref_unet, _ = accelerator.prepare(ref_unet, dummy_optimizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_steps_from_epochs:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    if accelerator.is_main_process:
        # TensorBoard hparams only accept scalars and strings.
        tracker_config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
        accelerator.init_trackers(
            args.project_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": os.path.basename(os.path.normpath(args.output_dir)),
                    "id": args.wandb_resume_id,
                    "resume": "allow" if args.wandb_resume_id else None,
                }
            },
        )

    validator = None
    if args.run_validation and accelerator.is_main_process:
        validator = Validator(args, accelerator, vae, weight_dtype, validation_prompts)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Model family = {args.model_family}")
    logger.info(f"  Num pairs = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimisation steps = {args.max_train_steps}")
    logger.info(f"  Pseudo-labels = {args.pseudo_label_path or 'none'}")

    global_step, first_epoch = 0, 0
    if args.resume_from_checkpoint:
        path = resolve_resume_path(args)
        if path is None:
            logger.info("No checkpoint found; starting a new run")
        else:
            logger.info(f"Resuming from {path}")
            accelerator.load_state(path)
            global_step = int(os.path.basename(os.path.normpath(path)).split("-")[1])
            first_epoch = global_step // steps_per_epoch

    progress = tqdm(
        range(args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process
    )

    if validator is not None and global_step == 0:
        validator.run(unet, step=0)
    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # [B, 2C, H, W] -> [2B, C, H, W] laid out as [winners; losers].
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                pixel_values = torch.cat(pixel_values.chunk(2, dim=1))
                latents = encode_images(vae, pixel_values, args.vae_encode_batch_size).to(weight_dtype)
                num_pairs = latents.shape[0] // 2

                # Winner and loser share the noise and the timestep.
                noise = torch.randn_like(latents[:num_pairs]).repeat(2, 1, 1, 1)
                timesteps, labels = sample_timesteps(args, batch, num_pairs, noise_scheduler, latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise_target(noise_scheduler, latents, noise, timesteps)

                cond = conditioner.encode(batch, args.resolution, weight_dtype).repeat(2)
                model_pred = unet(noisy_latents, timesteps, **cond.unet_kwargs()).sample
                with torch.no_grad():
                    if args.use_lora:
                        with adapters_disabled(unet):
                            ref_pred = unet(noisy_latents, timesteps, **cond.unet_kwargs()).sample
                    else:
                        # Match the policy, whose forward accelerate wraps in autocast.
                        with accelerator.autocast():
                            ref_pred = ref_unet(noisy_latents, timesteps, **cond.unet_kwargs()).sample

                model_losses = per_sample_mse(model_pred, target)
                ref_losses = per_sample_mse(ref_pred, target)

                robust_weight = None
                if args.loss_type in ("gce", "sce"):
                    if args.use_reward_scores:
                        scores = torch.stack([batch[c] for c in data_lib.REWARD_SCORE_COLUMNS], dim=1)
                        robust_weight = reward_consensus_weight(scores.to(latents.device))
                    else:
                        robust_weight = torch.ones(num_pairs, device=latents.device)
                    if args.truncate_timesteps > 0:
                        robust_weight = robust_weight * (timesteps[:num_pairs] < args.truncate_timesteps)
                in_robust_window = args.robust_loss_start_step <= global_step <= args.robust_loss_end_step

                with torch.autocast(accelerator.device.type, enabled=False):
                    out = dpo_loss(
                        model_losses,
                        ref_losses,
                        beta=args.beta_dpo,
                        loss_type=args.loss_type,
                        labels=labels,
                        robust_weight=robust_weight,
                        gce_q=args.gce_q if in_robust_window else 0.0,
                        sce_beta=args.sce_beta if in_robust_window else 0.0,
                        log_zero=args.log_zero,
                    )

                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    accelerator.log({"train/grad_norm": float(grad_norm)}, step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients and ema_unet is not None:
                    ema_unet.step(unet.parameters())

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                if global_step % args.checkpointing_steps == 0:
                    if args.save_training_state:
                        accelerator.save_state(save_dir)
                    save_unet(args, accelerator, unet, save_dir)
                    logger.info(f"Saved checkpoint to {save_dir}")

                if args.run_validation and global_step % args.validation_steps == 0:
                    if ema_unet is not None:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    if validator is not None:
                        checkpoint_dir = save_dir if os.path.isdir(save_dir) else None
                        validator.run(unet, step=global_step, checkpoint_dir=checkpoint_dir)
                    accelerator.wait_for_everyone()
                    if ema_unet is not None:
                        ema_unet.restore(unet.parameters())

                if global_step % args.clear_cache_steps == 0:
                    clear_cache(logger, f"Step {global_step}")

                if (
                    args.update_ref_model_steps
                    and global_step % args.update_ref_model_steps == 0
                    and global_step < args.max_train_steps
                ):
                    logger.info(f"Copying policy weights into the reference model at step {global_step}")
                    copy_policy_into_reference(accelerator, unet, ref_unet)

            logs = {
                "train/loss": out.loss.item(),
                "train/model_mse": model_losses.mean().item(),
                "train/ref_mse": ref_losses.mean().item(),
                "train/implicit_acc": out.implicit_acc.item(),
                "train/labelled_fraction": out.num_labelled.item() / num_pairs,
                "train/lr": lr_scheduler.get_last_lr()[0],
            }
            progress.set_postfix(**{k.split("/")[1]: round(v, 5) for k, v in logs.items()})
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        logger.info(f"Finished epoch {epoch}")
        if args.update_ref_model_per_epoch and not args.update_ref_model_steps and epoch < args.num_train_epochs - 1:
            logger.info(f"Copying policy weights into the reference model after epoch {epoch}")
            copy_policy_into_reference(accelerator, unet, ref_unet)

    # Final model -----------------------------------------------------------
    accelerator.wait_for_everyone()
    if ema_unet is not None:
        ema_unet.copy_to(unet.parameters())
    final_dir = os.path.join(args.output_dir, "final")
    save_unet(args, accelerator, unet, final_dir)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.push_to_hub:
        from huggingface_hub import create_repo, upload_folder

        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id
        upload_folder(repo_id=repo_id, folder_path=final_dir, commit_message="End of training", token=args.hub_token)

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
