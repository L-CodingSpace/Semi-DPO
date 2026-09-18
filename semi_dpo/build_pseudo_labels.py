#!/usr/bin/env python
"""Select high-confidence timestep pseudo-labels for the next training round.

Reads the confidence shards of ``compute_confidence.py`` and writes a json file
``{idx: {"timestep_50": -1|0|1, ...}}`` consumed by ``train.py --pseudo_label_path``.

Examples::

    # the 20% most confident pairs per timestep interval (paper default)
    python semi_dpo/build_pseudo_labels.py --confidence_dir runs/p1_conf \
        --clean_idx_path data/clean_idx.json --mode percentile --values 80 \
        --output runs/p1_pseudo_labels.json

    # a fixed number of pairs per anchor (t=50 ... t=950)
    ... --mode top_k --values 200000 200000 200000 200000 200000 200000 150000 150000 150000 150000
"""

from __future__ import annotations

import argparse
import json
import os

from semi_dpo import pseudo_labels as pl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confidence_dir", required=True, help="Output directory of compute_confidence.py.")
    parser.add_argument("--output", required=True, help="Pseudo-label json to write.")
    parser.add_argument("--mode", choices=pl.SELECTION_MODES, required=True)
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        required=True,
        help=f"One value, or one per anchor {list(pl.TIMESTEP_ANCHORS)}.",
    )
    parser.add_argument(
        "--clean_idx_path",
        default=None,
        help="Json list of consensus-clean pair ids; they are labelled +1 at every timestep.",
    )
    parser.add_argument(
        "--past_pseudo_labels",
        nargs="*",
        default=[],
        help="Pseudo-label files of earlier iterations; their pairs are ranked last.",
    )
    return parser.parse_args(argv)


def main(args):
    confidences = pl.load_confidences(args.confidence_dir)
    print(f"Loaded confidences for {len(confidences)} pairs")

    past = {}
    for path in args.past_pseudo_labels:
        past.update(pl.load_pseudo_labels(path))
    clean_ids = []
    if args.clean_idx_path:
        with open(args.clean_idx_path, "r", encoding="utf-8") as handle:
            clean_ids = [str(i) for i in json.load(handle)]

    labels, cutoffs = pl.select_pseudo_labels(confidences, args.mode, args.values, past, clean_ids)
    if not labels:
        raise SystemExit(
            "No pair received a non-zero label. All confidences may be zero, which happens when the "
            "checkpoint equals the reference model; otherwise loosen the selection."
        )

    print(f"{'anchor':>14} {'cutoff':>10} {'+1':>9} {'-1':>9} {'0':>9}")
    histogram = pl.label_histogram(labels)
    for key, counts in histogram.items():
        print(f"{key:>14} {cutoffs[key]:>10.4f} {counts[1]:>9} {counts[-1]:>9} {counts[0]:>9}")
    print(f"Kept {len(labels)} pairs ({len(clean_ids)} clean)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(labels, handle)
    stats = {
        "mode": args.mode,
        "values": args.values,
        "num_pairs": len(labels),
        "num_clean": len(clean_ids),
        "cutoffs": cutoffs,
        "histogram": {k: {str(label): n for label, n in v.items()} for k, v in histogram.items()},
    }
    with open(os.path.splitext(args.output)[0] + ".stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(parse_args())
