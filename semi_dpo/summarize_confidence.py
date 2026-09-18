#!/usr/bin/env python
"""Per-timestep accuracy and confidence statistics of a model.

Run ``compute_confidence.py`` on the held-out clean split first; the human
labels there are trusted, so the fraction of positive confidences is the
model's preference accuracy at each timestep (Table 7 of the paper).  The
percentiles of ``|confidence|`` are a starting point for the per-interval
thresholds of ``build_pseudo_labels.py --mode threshold``.

Writes ``summary.csv`` and ``summary.json`` (and ``accuracy.png`` when
matplotlib is installed) into ``--output_dir`` (default: the confidence dir).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from semi_dpo.pseudo_labels import TIMESTEP_ANCHORS, load_confidences, timestep_key


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confidence_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 75, 80, 87.5, 90])
    return parser.parse_args(argv)


def summarize(confidences, percentiles):
    rows = []
    for anchor in TIMESTEP_ANCHORS:
        key = timestep_key(anchor)
        values = np.array([row[key] for row in confidences.values() if key in row], dtype=np.float64)
        if values.size == 0:
            continue
        row = {
            "timestep": anchor,
            "num_pairs": int(values.size),
            "accuracy": float((values > 0).mean()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
        for p in percentiles:
            row[f"abs_p{p:g}"] = float(np.percentile(np.abs(values), p))
        rows.append(row)
    return rows


def plot(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["timestep"] for r in rows], [100 * r["accuracy"] for r in rows], marker="o")
    ax.set_xlabel("timestep")
    ax.set_ylabel("accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(args):
    output_dir = args.output_dir or args.confidence_dir
    os.makedirs(output_dir, exist_ok=True)
    rows = summarize(load_confidences(args.confidence_dir), args.percentiles)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = ["timestep", "pairs", "acc%", "mean"] + [f"|z| p{p:g}" for p in args.percentiles]
    print("  ".join(f"{h:>9}" for h in header))
    for r in rows:
        cells = [r["timestep"], r["num_pairs"], f"{100 * r['accuracy']:.1f}", f"{r['mean']:.3f}"]
        cells += [f"{r[f'abs_p{p:g}']:.3f}" for p in args.percentiles]
        print("  ".join(f"{c:>9}" for c in cells))

    if plot(rows, os.path.join(output_dir, "accuracy.png")):
        print(f"Saved plot to {os.path.join(output_dir, 'accuracy.png')}")


if __name__ == "__main__":
    main(parse_args())
