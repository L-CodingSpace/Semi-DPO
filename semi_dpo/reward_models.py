"""Reward models used for consensus filtering and evaluation.

Every model exposes ``score(prompt, images) -> list[float]`` with one score
per image.  :class:`RewardScorer` bundles all of them.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

CACHE_ROOT = os.path.expanduser(os.environ.get("SEMI_DPO_CACHE", "~/.cache/semi_dpo"))

ImageInput = Union[Image.Image, Sequence[Image.Image]]


def _as_list(images: ImageInput) -> List[Image.Image]:
    return [images] if isinstance(images, Image.Image) else list(images)


def _load_openai_clip(device):
    import clip

    model, preprocess = clip.load("ViT-L/14", device=device, jit=False, download_root=os.path.join(CACHE_ROOT, "clip"))
    if str(device) == "cpu":
        model.float()
    model.logit_scale.requires_grad_(False)
    model.eval()
    return model, preprocess


class MLP(nn.Module):
    """Linear head of the LAION aesthetic predictor and ImageReward."""

    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CLIPScore(nn.Module):
    """Cosine similarity between CLIP ViT-L/14 text and image embeddings."""

    def __init__(self, device="cpu"):
        super().__init__()
        import clip

        self._tokenize = clip.tokenize
        self.device = device
        self.clip_model, self.preprocess = _load_openai_clip(device)

    @torch.no_grad()
    def score(self, prompt: str, images: ImageInput) -> List[float]:
        text = self._tokenize(prompt, truncate=True).to(self.device)
        text_features = F.normalize(self.clip_model.encode_text(text))
        pixels = torch.stack([self.preprocess(x) for x in _as_list(images)]).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(pixels))
        return (text_features * image_features).sum(dim=1).float().cpu().tolist()


class AestheticScore(nn.Module):
    """LAION improved aesthetic predictor (CLIP ViT-L/14 + MLP)."""

    URL = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
        "sac%2Blogos%2Bava1-l14-linearMSE.pth"
    )

    def __init__(self, device="cpu"):
        super().__init__()
        from ImageReward.utils import _download

        self.device = device
        self.clip_model, self.preprocess = _load_openai_clip(device)
        self.mlp = MLP(768)
        weights = _download(url=self.URL, root=os.path.join(CACHE_ROOT, "aesthetic"))
        self.mlp.load_state_dict(torch.load(weights, map_location=device), strict=False)
        self.mlp.to(device).eval()

    @torch.no_grad()
    def score(self, prompt: str, images: ImageInput) -> List[float]:
        pixels = torch.stack([self.preprocess(x) for x in _as_list(images)]).to(self.device)
        features = F.normalize(self.clip_model.encode_image(pixels)).float()
        return self.mlp(features).flatten().float().cpu().tolist()


class ImageReward(nn.Module):
    """ImageReward (BLIP + MLP), normalised as in the official release."""

    MEAN = 0.16717362830052426
    STD = 1.0333394966054072

    def __init__(self, device="cpu"):
        super().__init__()
        from ImageReward import BLIP_Pretrain, ImageReward_download

        self.device = device
        root = os.path.join(CACHE_ROOT, "ImageReward")
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", root)
        weights = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt", root)

        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=med_config)
        self.mlp = MLP(768)
        self.load_state_dict(torch.load(weights, map_location=device), strict=False)
        self.preprocess = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.to(device).eval()

    @torch.no_grad()
    def score(self, prompt: str, images: ImageInput) -> List[float]:
        text = self.blip.tokenizer(
            prompt, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(self.device)
        pixels = torch.stack([self.preprocess(x) for x in _as_list(images)]).to(self.device)
        image_embeds = self.blip.visual_encoder(pixels)
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=self.device)
        output = self.blip.text_encoder(
            text.input_ids.expand(len(pixels), -1),
            attention_mask=text.attention_mask.expand(len(pixels), -1),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        rewards = self.mlp(output.last_hidden_state[:, 0, :].float())
        return ((rewards - self.MEAN) / self.STD).flatten().float().cpu().tolist()


class PickScore(nn.Module):
    """PickScore v1 (CLIP ViT-H/14 fine-tuned on Pick-a-Pic)."""

    def __init__(
        self,
        processor="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model="yuvalkirstain/PickScore_v1",
        device="cpu",
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor)
        self.model = AutoModel.from_pretrained(model).to(device).eval()

    @torch.no_grad()
    def score(self, prompt: str, images: ImageInput) -> List[float]:
        common = dict(padding=True, truncation=True, max_length=77, return_tensors="pt")
        image_inputs = self.processor(images=_as_list(images), **common).to(self.device)
        text_inputs = self.processor(text=prompt, **common).to(self.device)
        image_embs = F.normalize(self.model.get_image_features(**image_inputs), dim=-1)
        text_embs = F.normalize(self.model.get_text_features(**text_inputs), dim=-1)
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        return scores.float().cpu().tolist()


class HPSv2(nn.Module):
    """Human Preference Score v2 / v2.1."""

    def __init__(self, version="v2.1", device="cpu"):
        super().__init__()
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from huggingface_hub import hf_hub_download

        if version not in ("v2", "v2.1"):
            raise ValueError(f"Invalid HPS version: {version}")
        self.device = device
        self.model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint = torch.load(hf_hub_download("xswu/HPSv2", f"HPS_{version}_compressed.pt"), map_location=device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model.to(device).eval()

    @torch.no_grad()
    def score(self, prompt: str, images: ImageInput) -> List[float]:
        pixels = torch.stack([self.preprocess(x) for x in _as_list(images)]).to(self.device)
        text = self.tokenizer([prompt]).to(self.device, non_blocking=True)
        outputs = self.model(pixels, text)
        return (outputs["image_features"] @ outputs["text_features"].T).flatten().float().cpu().tolist()


REWARD_MODELS = {
    "clip_score": CLIPScore,
    "aesthetic_score": AestheticScore,
    "image_reward_score": ImageReward,
    "pick_score": PickScore,
    "hps_score": HPSv2,
}


class RewardScorer:
    """Loads a set of reward models and scores images with all of them."""

    def __init__(self, names: Iterable[str] = tuple(REWARD_MODELS), device="cpu"):
        unknown = set(names) - set(REWARD_MODELS)
        if unknown:
            raise ValueError(f"Unknown reward models {sorted(unknown)}; choose from {list(REWARD_MODELS)}")
        self.models = {name: REWARD_MODELS[name](device=device) for name in names}

    def score(self, prompt: str, images: ImageInput) -> Dict[str, List[float]]:
        return {name: model.score(prompt, images) for name, model in self.models.items()}
