#!/usr/bin/env python
"""Generate images for a prompt set and score them with the reward models.

Writes ``{prompt_idx}-{image_idx}.png`` and a matching ``.json`` with the five
reward scores to ``--output_dir``; already finished prompts are skipped, so
the script can be resumed.  Summarise and compare runs with ``report.py``.

    accelerate launch semi_dpo/evaluate.py --model_family sdxl \
        --checkpoint runs/sdxl_p1/checkpoint-2000 \
        --dataset_name yuvalkirstain/pickapic_v2 --dataset_split test_unique \
        --output_dir eval/sdxl_p1_2000/pickapic

Without ``--checkpoint`` the base model is evaluated.  ``--checkpoint`` also
accepts hub ids with a ``unet`` folder, e.g. ``mhdang/dpo-sdxl-text2image-v1``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from accelerate import PartialState
from tqdm.auto import tqdm

from semi_dpo.data import load_split, prompt_column
from semi_dpo.models import FAMILIES, build_pipeline, load_unet
from semi_dpo.reward_models import REWARD_MODELS, RewardScorer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint or hub model with a unet folder.")
    parser.add_argument("--lora", default=None, help="Directory with pytorch_lora_weights.safetensors.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=None, help="Defaults to the pipeline default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward_models", nargs="+", default=list(REWARD_MODELS), choices=list(REWARD_MODELS))
    parser.add_argument("--no_save_images", action="store_true")
    args = parser.parse_args(argv)
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or FAMILIES[args.model_family].default_model
    return args


@torch.no_grad()
def main(args):
    state = PartialState()
    device = state.device
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    unet = load_unet(args.pretrained_model_name_or_path, args.checkpoint).to(dtype)
    pipeline = build_pipeline(args.model_family, args.pretrained_model_name_or_path, unet=unet, dtype=dtype)
    if args.lora:
        pipeline.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipeline.to(device)
    scorer = RewardScorer(args.reward_models, device=device)

    dataset = load_split(args.dataset_name, args.dataset_split)
    if args.max_prompts is not None:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))
    prompts = list(dataset[prompt_column(dataset)])
    os.makedirs(args.output_dir, exist_ok=True)

    def done(i):
        return os.path.exists(os.path.join(args.output_dir, f"{i}-{args.num_images_per_prompt - 1}.json"))

    call_kwargs = {"num_inference_steps": args.num_inference_steps, "num_images_per_prompt": args.num_images_per_prompt}
    if args.guidance_scale is not None:
        call_kwargs["guidance_scale"] = args.guidance_scale

    todo = [i for i in range(len(prompts)) if not done(i)]
    with state.split_between_processes(todo) as indices:
        for i in tqdm(indices, disable=not state.is_local_main_process):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            images = pipeline(prompts[i], generator=generator, **call_kwargs).images
            for k, image in enumerate(images):
                stem = os.path.join(args.output_dir, f"{i}-{k}")
                if not args.no_save_images:
                    image.save(stem + ".png")
                scores = {name: values[0] for name, values in scorer.score(prompts[i], image).items()}
                with open(stem + ".json", "w", encoding="utf-8") as handle:
                    json.dump({"prompt": prompts[i], **scores}, handle, indent=2)
    state.wait_for_everyone()


if __name__ == "__main__":
    main(parse_args())
