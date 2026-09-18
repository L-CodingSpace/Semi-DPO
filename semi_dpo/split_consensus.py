#!/usr/bin/env python
"""Multi-reward consensus: split a scored pair dataset into clean and noisy pairs.

A pair is *clean* when every reward model prefers the human winner, i.e.
``r_k(x_w) - r_k(x_l) > 0`` for all ``k`` (Sec. 3.3 of the paper); all other
pairs are *noisy* and later receive pseudo-labels.  ``--criterion models_agree``
instead only requires the reward models to agree with each other, regardless
of the human label.

Outputs in ``--output_dir``:

* ``clean/``: a ``DatasetDict`` with ``train`` and a held-out ``test`` split.
  Train stage 1 on ``train``; use ``test`` to measure per-timestep accuracy.
* ``clean_idx.json``: ``idx`` of the clean *train* pairs (for build_pseudo_labels.py).
* ``stats.json``: split sizes and per-model agreement rates.
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import DatasetDict

from semi_dpo.data import REWARD_SCORE_COLUMNS, load_split


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", required=True, help="Dataset with *_score columns (score_pairs.py).")
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_models", nargs="+", default=list(REWARD_SCORE_COLUMNS))
    parser.add_argument("--criterion", choices=["human_agree", "models_agree"], default="human_agree")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.02,
        help="Fraction (<1) or number (>=1) of clean pairs held out for accuracy analysis.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8)
    return parser.parse_args(argv)


def human_winner(example) -> int:
    if "human_scores" in example:
        scores = example["human_scores"]
        return 0 if scores[0] >= scores[1] else 1
    if "winner_idx" in example:
        return int(example["winner_idx"])
    return 0


TIE_TOLERANCE = 1e-6


def reward_votes(example, reward_models):
    """Index of the image each reward model prefers, or None on a tie."""
    votes = []
    for name in reward_models:
        s0, s1 = example[name]
        votes.append(None if abs(s0 - s1) <= TIE_TOLERANCE else (0 if s0 > s1 else 1))
    return votes


def is_clean(example, reward_models, criterion) -> bool:
    votes = reward_votes(example, reward_models)
    if criterion == "human_agree":
        winner = human_winner(example)
        return all(vote == winner for vote in votes)
    decided = [vote for vote in votes if vote is not None]
    return bool(decided) and len(set(decided)) == 1


def main(args):
    dataset = load_split(args.dataset_name, args.dataset_split)
    missing = [name for name in args.reward_models + ["idx"] if name not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset is missing columns {missing}; run score_pairs.py first")

    columns = [c for c in args.reward_models + ["human_scores", "winner_idx"] if c in dataset.column_names]
    flags = dataset.select_columns(columns).map(
        lambda ex: {"clean": is_clean(ex, args.reward_models, args.criterion)},
        num_proc=args.num_proc,
        desc="Checking consensus",
    )["clean"]
    clean = dataset.select([i for i, flag in enumerate(flags) if flag])

    test_size = int(args.test_size) if args.test_size >= 1 else args.test_size
    split = clean.train_test_split(test_size=test_size, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict(train=split["train"], test=split["test"]).save_to_disk(os.path.join(args.output_dir, "clean"))
    with open(os.path.join(args.output_dir, "clean_idx.json"), "w", encoding="utf-8") as handle:
        json.dump([int(i) for i in split["train"]["idx"]], handle)

    agreement = {}
    scored = dataset.select_columns(columns)
    for name in args.reward_models:
        agree = scored.map(
            lambda ex, name=name: {"agree": reward_votes(ex, [name])[0] == human_winner(ex)},
            num_proc=args.num_proc,
            desc=f"Agreement of {name}",
        )["agree"]
        agreement[name] = sum(agree) / len(agree)

    stats = {
        "criterion": args.criterion,
        "reward_models": args.reward_models,
        "total": len(dataset),
        "clean": len(clean),
        "clean_fraction": len(clean) / len(dataset),
        "clean_train": len(split["train"]),
        "clean_test": len(split["test"]),
        "agreement_with_human": agreement,
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(parse_args())
