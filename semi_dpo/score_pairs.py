#!/usr/bin/env python
"""Score both images of every pair with the reward models.

Scores are written per process to ``<work_dir>/scores_rank{r}.jsonl`` so an
interrupted run can resume.  Once every pair is scored, the scores are added
to the dataset as ``[score_0, score_1]`` columns (``clip_score``,
``aesthetic_score``, ``image_reward_score``, ``pick_score``, ``hps_score``)
and the result is saved to ``--output_dir``.

    accelerate launch semi_dpo/score_pairs.py \
        --dataset_name data/pickapic_v2 --work_dir runs/scores --output_dir data/pickapic_v2_scored
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from accelerate import PartialState
from diffusers.utils import load_image
from tqdm.auto import tqdm

from semi_dpo.data import load_split, prompt_column
from semi_dpo.reward_models import REWARD_MODELS, RewardScorer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--work_dir", required=True, help="Where the per-process score shards are written.")
    parser.add_argument("--output_dir", required=True, help="Where the scored dataset is saved.")
    parser.add_argument("--reward_models", nargs="+", default=list(REWARD_MODELS), choices=list(REWARD_MODELS))
    return parser.parse_args(argv)


def read_shards(work_dir):
    scores = {}
    for path in glob.glob(os.path.join(work_dir, "scores_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    scores[row.pop("row")] = row
    return scores


def main(args):
    state = PartialState()
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = load_split(args.dataset_name, args.dataset_split)
    prompts = prompt_column(dataset)

    done = read_shards(args.work_dir)
    todo = [i for i in range(len(dataset)) if i not in done]
    if state.is_main_process:
        print(f"{len(done)} pairs already scored, {len(todo)} to go")

    if todo:
        scorer = RewardScorer(args.reward_models, device=state.device)
        shard_path = os.path.join(args.work_dir, f"scores_rank{state.process_index:03d}.jsonl")
        with state.split_between_processes(todo) as rows, open(shard_path, "a", encoding="utf-8") as shard:
            for row in tqdm(rows, disable=not state.is_local_main_process):
                example = dataset[row]
                images = [load_image(image).convert("RGB") for image in example["images"]]
                shard.write(json.dumps({"row": row, **scorer.score(example[prompts], images)}) + "\n")
                shard.flush()
    state.wait_for_everyone()

    if state.is_main_process:
        scores = read_shards(args.work_dir)
        missing = len(dataset) - len(scores)
        if missing:
            raise RuntimeError(f"{missing} pairs have no scores; rerun to finish them")
        for name in args.reward_models:
            column = [scores[i][name] for i in range(len(dataset))]
            if name in dataset.column_names:
                dataset = dataset.remove_columns(name)
            dataset = dataset.add_column(name, column)
        dataset.save_to_disk(args.output_dir)
        print(f"Saved scored dataset to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
