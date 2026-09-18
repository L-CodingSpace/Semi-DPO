"""Model loading and text conditioning for SD 1.5 and SDXL.

Everything that differs between the two model families lives here, so the
training and pseudo-labelling loops can be written once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    default_model: str
    default_resolution: int
    # SDXL's VAE overflows in half precision, so it is always kept in fp32.
    vae_in_fp32: bool
    # SDXL conditions on crop coordinates, so its pipeline never flips images.
    supports_hflip: bool
    validation_steps: int
    validation_guidance: Optional[float]


FAMILIES: Dict[str, FamilySpec] = {
    "sd15": FamilySpec(
        name="sd15",
        default_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        default_resolution=512,
        vae_in_fp32=False,
        supports_hflip=True,
        validation_steps=50,
        validation_guidance=None,
    ),
    "sdxl": FamilySpec(
        name="sdxl",
        default_model="stabilityai/stable-diffusion-xl-base-1.0",
        default_resolution=1024,
        vae_in_fp32=True,
        supports_hflip=False,
        validation_steps=25,
        validation_guidance=5.0,
    ),
}


@dataclass
class Conditioning:
    """Text conditioning for one forward pass of the UNet."""

    encoder_hidden_states: torch.Tensor
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None

    def repeat(self, n: int) -> "Conditioning":
        hidden = self.encoder_hidden_states.repeat(n, 1, 1)
        added = None
        if self.added_cond_kwargs is not None:
            added = {k: v.repeat(n, 1) for k, v in self.added_cond_kwargs.items()}
        return Conditioning(hidden, added)

    def unet_kwargs(self) -> Dict:
        kwargs = {"encoder_hidden_states": self.encoder_hidden_states}
        if self.added_cond_kwargs is not None:
            kwargs["added_cond_kwargs"] = self.added_cond_kwargs
        return kwargs


def _text_encoder_class(model_path: str, subfolder: str, revision: Optional[str]):
    config = PretrainedConfig.from_pretrained(model_path, subfolder=subfolder, revision=revision)
    architecture = config.architectures[0]
    if architecture == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported text encoder: {architecture}")


class PromptTokenizer:
    """Tokenizes prompts for every text encoder into ``input_ids_{i}`` tensors.

    Kept separate from the text encoders so dataloader workers only receive
    the (small) tokenizers.
    """

    def __init__(self, tokenizers: Sequence):
        self.tokenizers = list(tokenizers)
        self.keys = [f"input_ids_{i}" for i in range(len(self.tokenizers))]

    def __call__(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return {
            key: tokenizer(
                list(prompts),
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            for key, tokenizer in zip(self.keys, self.tokenizers)
        }


class TextConditioner:
    """Tokenizers and frozen text encoders of a model family.

    SD 1.5 has one CLIP text encoder.  SDXL has two, concatenates their
    penultimate hidden states and additionally conditions on the pooled
    embedding of the second encoder plus the image size / crop ``time_ids``.
    """

    def __init__(self, family: str, model_path: str, revision: Optional[str] = None, variant: Optional[str] = None):
        self.family = family
        self.is_sdxl = family == "sdxl"
        subfolders: Sequence[Tuple[str, str]] = [("tokenizer", "text_encoder")]
        if self.is_sdxl:
            subfolders = [("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2")]

        tokenizers = []
        self.text_encoders = []
        for tokenizer_dir, encoder_dir in subfolders:
            tokenizers.append(
                AutoTokenizer.from_pretrained(model_path, subfolder=tokenizer_dir, revision=revision, use_fast=False)
            )
            encoder_cls = _text_encoder_class(model_path, encoder_dir, revision)
            encoder = encoder_cls.from_pretrained(model_path, subfolder=encoder_dir, revision=revision, variant=variant)
            encoder.requires_grad_(False)
            self.text_encoders.append(encoder)
        self.tokenize = PromptTokenizer(tokenizers)

    @property
    def input_keys(self) -> List[str]:
        return self.tokenize.keys

    def to(self, device, dtype=None) -> "TextConditioner":
        for encoder in self.text_encoders:
            encoder.to(device, dtype=dtype)
        return self

    @torch.no_grad()
    def encode(self, batch: Dict, resolution: int, dtype: torch.dtype) -> Conditioning:
        """Encode the tokenized prompts of ``batch`` (one entry per pair)."""
        if not self.is_sdxl:
            encoder = self.text_encoders[0]
            hidden = encoder(batch["input_ids_0"].to(encoder.device))[0]
            return Conditioning(hidden)

        hidden_states = []
        pooled = None
        for key, encoder in zip(self.input_keys, self.text_encoders):
            output = encoder(batch[key].to(encoder.device), output_hidden_states=True)
            # Only the pooled output of the last encoder is used.
            pooled = output[0]
            hidden_states.append(output.hidden_states[-2])
        hidden = torch.cat(hidden_states, dim=-1)
        pooled = pooled.view(hidden.shape[0], -1)

        device = hidden.device
        time_ids = torch.tensor(
            [
                list(original_size) + list(crop_top_left) + [resolution, resolution]
                for original_size, crop_top_left in zip(batch["original_sizes"], batch["crop_top_lefts"])
            ],
            device=device,
            dtype=dtype,
        )
        return Conditioning(hidden, {"time_ids": time_ids, "text_embeds": pooled})


def load_unet(
    base_model: str,
    checkpoint: Optional[str] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> UNet2DConditionModel:
    """Load a UNet from the base model or from a training checkpoint.

    ``checkpoint`` may be a checkpoint directory written by ``semi_dpo.train``
    (holding ``unet/unet.pt`` from FSDP, a diffusers ``unet/`` folder or LoRA
    weights, which are merged into the base UNet), a diffusers UNet folder
    itself, or any diffusers model path / hub id with a ``unet`` subfolder.
    """
    if checkpoint is None:
        return UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", revision=revision, variant=variant)

    lora_path = os.path.join(checkpoint, LORA_WEIGHT_NAME)
    if os.path.isfile(lora_path):
        # LoRA run: base UNet with the adapter merged in.
        from diffusers import StableDiffusionPipeline

        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", revision=revision, variant=variant)
        state_dict, alphas = StableDiffusionPipeline.lora_state_dict(checkpoint, weight_name=LORA_WEIGHT_NAME)
        StableDiffusionPipeline.load_lora_into_unet(state_dict, network_alphas=alphas, unet=unet)
        unet.fuse_lora()
        unet.unload_lora()
        return unet

    for state_dict_path in (os.path.join(checkpoint, "unet", "unet.pt"), os.path.join(checkpoint, "unet.pt")):
        if os.path.isfile(state_dict_path):
            unet = UNet2DConditionModel.from_pretrained(
                base_model, subfolder="unet", revision=revision, variant=variant
            )
            unet.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True), strict=True)
            return unet
    if os.path.isfile(os.path.join(checkpoint, "config.json")):
        return UNet2DConditionModel.from_pretrained(checkpoint)
    return UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")


def load_vae(base_model: str, vae_path: Optional[str] = None, revision=None, variant=None) -> AutoencoderKL:
    if vae_path is None:
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", revision=revision, variant=variant)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)
    vae.requires_grad_(False)
    return vae


def load_noise_scheduler(base_model: str) -> DDPMScheduler:
    return DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")


def enforce_zero_terminal_snr(scheduler: DDPMScheduler) -> None:
    """Rescale the noise schedule to zero terminal SNR (https://arxiv.org/abs/2305.08891).

    Required when fine-tuning SDXL-Turbo.
    """
    alphas_bar_sqrt = (1 - scheduler.betas).cumprod(0).sqrt()
    first, last = alphas_bar_sqrt[0].clone(), alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= last
    alphas_bar_sqrt *= first / (first - last)
    alphas_bar = alphas_bar_sqrt**2
    alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)


@torch.no_grad()
def encode_images(vae: AutoencoderKL, pixel_values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """VAE-encode images in chunks and scale the latents."""
    latents = [
        vae.encode(pixel_values[i : i + batch_size]).latent_dist.sample()
        for i in range(0, pixel_values.shape[0], batch_size)
    ]
    return torch.cat(latents) * vae.config.scaling_factor


def noise_target(scheduler: DDPMScheduler, latents, noise, timesteps) -> torch.Tensor:
    if scheduler.config.prediction_type == "epsilon":
        return noise
    if scheduler.config.prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")


def build_pipeline(
    family: str,
    base_model: str,
    unet: UNet2DConditionModel,
    vae: Optional[AutoencoderKL] = None,
    dtype: torch.dtype = torch.float32,
    revision=None,
    variant=None,
) -> DiffusionPipeline:
    """Text-to-image pipeline around a (trained) UNet with the base model's default scheduler."""
    kwargs = {"unet": unet, "torch_dtype": dtype, "revision": revision, "variant": variant}
    if vae is not None:
        kwargs["vae"] = vae
    if family == "sd15":
        kwargs["safety_checker"] = None
    pipeline = DiffusionPipeline.from_pretrained(base_model, **kwargs)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def use_dpm_solver(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline
