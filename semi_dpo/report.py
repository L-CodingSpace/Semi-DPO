#!/usr/bin/env python
"""Summarise ``evaluate.py`` runs and compare them against a baseline.

For every run directory the mean and std of each reward score is reported.
With ``--baseline``, the win rate is the share of images that score higher
than the baseline image generated for the same prompt and seed.

    python semi_dpo/report.py eval/semi_dpo/pickapic eval/dpo/pickapic \
        --baseline eval/sd15_base/pickapic --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict

import numpy as np

from semi_dpo.reward_models import REWARD_MODELS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Output directories of evaluate.py.")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output", default=None, help="Optional csv file for the table.")
    return parser.parse_args(argv)


def load_run(directory: str) -> Dict[str, Dict[str, float]]:
    scores = {}
    for path in glob.glob(os.path.join(directory, "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            row = json.load(handle)
        scores[os.path.basename(path)] = {k: float(v) for k, v in row.items() if k in REWARD_MODELS}
    if not scores:
        raise FileNotFoundError(f"No score files in {directory}")
    return scores


def summarise(run, baseline=None):
    metrics = sorted({metric for row in run.values() for metric in row})
    summary = {"num_images": len(run)}
    for metric in metrics:
        values = np.array([row[metric] for row in run.values() if metric in row])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
        if baseline is not None:
            shared = [name for name in run if name in baseline and metric in baseline[name]]
            if shared:
                wins = np.mean([run[n][metric] > baseline[n][metric] for n in shared])
                summary[f"{metric}_win_rate"] = float(wins)
    return summary


def main(args):
    baseline = load_run(args.baseline) if args.baseline else None
    table = []
    for directory in args.runs:
        table.append({"run": directory, **summarise(load_run(directory), baseline)})

    metrics = [m for m in REWARD_MODELS if f"{m}_mean" in table[0]]
    header = f"{'run':<40}" + "".join(f"{m:>20}" for m in metrics)
    print(header)
    for row in table:
        cells = []
        for metric in metrics:
            cell = f"{row[f'{metric}_mean']:.4f}"
            if f"{metric}_win_rate" in row:
                cell += f" ({100 * row[f'{metric}_win_rate']:.1f}%)"
            cells.append(f"{cell:>20}")
        print(f"{row['run'][-40:]:<40}" + "".join(cells))
    if baseline is not None:
        print(f"\nWin rates in parentheses are against {args.baseline}.")

    if args.output:
        fields = sorted({key for row in table for key in row}, key=lambda k: (k != "run", k))
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(parse_args())
