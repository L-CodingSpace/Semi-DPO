#!/usr/bin/env python
"""Convert Pick-a-Pic into the pair format used by this repository.

Ties (``label_0 == 0.5``) are removed, which leaves the 851,293 training pairs
of Pick-a-Pic v2 also used by Diffusion-DPO.  Output columns:

* ``idx``: position in the tie-free dataset, used to key pseudo-labels.
* ``prompt``: the caption.
* ``images``: ``[image_0, image_1]``.
* ``human_scores``: ``[label_0, label_1]``.
"""

from __future__ import annotations

import argparse

from datasets import Image, Sequence, load_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", default="yuvalkirstain/pickapic_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args(argv)


def to_pairs(batch):
    return {
        "prompt": batch["caption"],
        "images": [
            [{"bytes": a, "path": None}, {"bytes": b, "path": None}] for a, b in zip(batch["jpg_0"], batch["jpg_1"])
        ],
        "human_scores": [[a, b] for a, b in zip(batch["label_0"], batch["label_1"])],
    }


def main(args):
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir, num_proc=args.num_proc)
    dataset = dataset.filter(lambda label: label != 0.5, input_columns="label_0", num_proc=args.num_proc)
    dataset = dataset.map(
        to_pairs, batched=True, remove_columns=dataset.column_names, num_proc=args.num_proc, desc="Formatting pairs"
    )
    dataset = dataset.cast_column("images", Sequence(Image()))
    dataset = dataset.add_column("idx", list(range(len(dataset))))
    dataset.save_to_disk(args.output_dir)
    print(f"Saved {len(dataset)} pairs to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
