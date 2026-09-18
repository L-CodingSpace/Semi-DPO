"""Preference-pair datasets.

Expected columns of a pair dataset (a 🤗 ``datasets`` folder, hub id or json):

* ``images``: two images (PIL images, paths or URLs).
* ``prompt`` (or ``caption``): the text prompt.
* ``human_scores`` (``[s0, s1]``) or ``winner_idx`` (``0``/``1``).  Without
  either, the first image is taken as the winner.
* ``idx``: a stable integer id, required for pseudo-labels.
* Optional ``*_score`` columns (``[s0, s1]``) from the reward models.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional, Sequence

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from diffusers.utils import load_image
from torchvision import transforms

REWARD_SCORE_COLUMNS = ("clip_score", "aesthetic_score", "image_reward_score", "pick_score", "hps_score")


def load_split(name_or_path: str, split: Optional[str] = None, cache_dir: Optional[str] = None) -> Dataset:
    """Load one split from a local folder, a json file or the Hugging Face hub."""
    if os.path.exists(name_or_path):
        if name_or_path.endswith((".json", ".jsonl")):
            dataset = load_dataset("json", data_files=name_or_path)
        else:
            dataset = load_from_disk(name_or_path)
    else:
        dataset = load_dataset(name_or_path, cache_dir=cache_dir)

    if isinstance(dataset, DatasetDict):
        if split is None:
            split = "train" if "train" in dataset else next(iter(dataset))
        dataset = dataset[split]
    return dataset


def prompt_column(dataset: Dataset) -> str:
    for column in ("prompt", "caption", "Prompt"):
        if column in dataset.column_names:
            return column
    raise KeyError(f"No prompt column in {dataset.column_names}")


def select_indices(dataset: Dataset, keep_ids: Sequence[str]) -> Dataset:
    """Keep the rows whose ``idx`` column is in ``keep_ids`` (compared as strings)."""
    wanted = set(map(str, keep_ids))
    positions = [pos for pos, idx in enumerate(dataset["idx"]) if str(idx) in wanted]
    return dataset.select(positions)


def image_transforms(resolution: int, random_crop: bool = False, hflip: bool = False) -> Callable:
    steps = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(resolution) if random_crop else transforms.CenterCrop(resolution),
    ]
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(steps)


def _human_scores(examples: Dict, i: int, num_images: int) -> List[float]:
    if "human_scores" in examples:
        return list(examples["human_scores"][i])
    if "winner_idx" in examples:
        return [1.0, 0.0] if examples["winner_idx"][i] == 0 else [0.0, 1.0]
    return list(range(num_images - 1, -1, -1))


def preprocess_pairs(
    examples: Dict,
    transform: Callable,
    tokenize: Callable[[Sequence[str]], Dict[str, torch.Tensor]],
    pseudo_labels: Optional[Dict[str, Dict[str, int]]] = None,
    random_swap_prob: float = 0.0,
) -> Dict:
    """Turn a batch of rows into ``[winner, loser]`` training pairs.

    The two images are concatenated along the channel axis, winner first.
    ``random_swap_prob`` randomly flips labels (a noisy-label ablation); rows
    with a ``neg_prompt`` column swap images and prompt with probability 0.5.
    """
    prompts_key = "prompt" if "prompt" in examples else "caption"
    score_keys = [key for key in examples if key.endswith("_score")]

    out: Dict[str, List] = {"pixel_values": [], "prompt": [], "original_sizes": [], "crop_top_lefts": []}
    out.update({key: [] for key in score_keys})
    if pseudo_labels is not None:
        out["pseudo_labels"] = []

    for i, images in enumerate(examples["images"]):
        prompt = examples[prompts_key][i]
        scores = _human_scores(examples, i, len(images))
        win, lose = (0, 1) if scores[0] >= scores[1] else (1, 0)

        if "neg_prompt" in examples and random.random() > 0.5:
            win, lose = lose, win
            prompt = examples["neg_prompt"][i]
        if random.random() < random_swap_prob:
            win, lose = lose, win

        image_win = load_image(images[win]).convert("RGB")
        image_lose = load_image(images[lose]).convert("RGB")

        out["pixel_values"].append(torch.cat([transform(image_win), transform(image_lose)], dim=0))
        out["prompt"].append(prompt)
        out["original_sizes"].append((image_win.height, image_win.width))
        # Images are resized and center-cropped to the training resolution.
        out["crop_top_lefts"].append((0, 0))
        for key in score_keys:
            out[key].append([examples[key][i][win], examples[key][i][lose]])
        if pseudo_labels is not None:
            out["pseudo_labels"].append(pseudo_labels.get(str(examples["idx"][i]), {}))

    out.update(tokenize(out["prompt"]))
    if "idx" in examples:
        out["idx"] = list(examples["idx"])
    return out


def collate_pairs(examples: List[Dict]) -> Dict:
    batch = {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]).contiguous().float(),
        "original_sizes": [e["original_sizes"] for e in examples],
        "crop_top_lefts": [e["crop_top_lefts"] for e in examples],
    }
    for key in examples[0]:
        if key.startswith("input_ids_"):
            batch[key] = torch.stack([e[key] for e in examples])
        elif key.endswith("_score"):
            batch[key] = torch.tensor([e[key] for e in examples])
    if "pseudo_labels" in examples[0]:
        batch["pseudo_labels"] = [e["pseudo_labels"] for e in examples]
    if "idx" in examples[0]:
        batch["idx"] = torch.tensor([int(e["idx"]) for e in examples])
    return batch
