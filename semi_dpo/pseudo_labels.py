"""Timestep-conditional pseudo-labels.

A pseudo-label file maps a dataset index to one label per timestep anchor::

    {"12345": {"timestep_50": 1, "timestep_150": 0, ..., "timestep_950": -1}}

``+1`` keeps the human preference for that timestep, ``-1`` flips it and ``0``
marks the pair as unlabelled there.  Every anchor ``t`` stands for the interval
``[t - 50, t + 50)``, so the ten anchors tile the full diffusion schedule.

This module deliberately avoids torch so that the label logic can be unit
tested on its own.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

import numpy as np

#: Timestep anchors used throughout the pipeline.
TIMESTEP_ANCHORS: Sequence[int] = (50, 150, 250, 350, 450, 550, 650, 750, 850, 950)

#: Half width of the interval each anchor represents.
INTERVAL_HALF_WIDTH = 50

#: Number of training timesteps of the diffusion schedule.
MAX_TIMESTEP = 1000

PseudoLabelDict = Dict[str, Dict[str, int]]


class Interval(NamedTuple):
    """A half-open timestep range ``[lower, upper)`` carrying one label."""

    lower: int
    upper: int
    anchor: int
    label: int

    @property
    def size(self) -> int:
        return self.upper - self.lower


def timestep_key(anchor: int) -> str:
    return f"timestep_{anchor}"


def label_from_confidence(confidence: float, tau: float) -> int:
    """Threshold a preference logit into ``{-1, 0, +1}``.

    ``confidence`` is ``beta * (ref_diff - model_diff)``: positive when the
    model agrees with the human label, negative when it disagrees.  Pairs whose
    magnitude stays below ``tau`` are left unlabelled.
    """
    if confidence > tau:
        return 1
    if confidence < -tau:
        return -1
    return 0


def extract_valid_intervals(
    labels: Dict[str, int],
    max_timestep: int = MAX_TIMESTEP,
    half_width: int = INTERVAL_HALF_WIDTH,
) -> List[Interval]:
    """Turn one sample's per-anchor labels into disjoint half-open intervals.

    Anchors labelled ``0`` are dropped.  With the default anchors the
    intervals tile ``[0, max_timestep)`` exactly; custom anchors closer than
    ``2 * half_width`` are clipped so that no timestep belongs to two labels.
    """
    intervals: List[Interval] = []
    for key, label in labels.items():
        if label == 0 or not key.startswith("timestep_"):
            continue
        anchor = int(key.split("_")[1])
        lower = max(0, anchor - half_width)
        upper = min(max_timestep, anchor + half_width)
        if upper > lower:
            intervals.append(Interval(lower, upper, anchor, int(label)))

    intervals.sort(key=lambda interval: interval.lower)

    disjoint: List[Interval] = []
    for interval in intervals:
        if disjoint and interval.lower < disjoint[-1].upper:
            interval = interval._replace(lower=disjoint[-1].upper)
        if interval.size > 0:
            disjoint.append(interval)
    return disjoint


def sample_timestep(
    intervals: Sequence[Interval],
    rng: Optional[random.Random] = None,
    max_timestep: int = MAX_TIMESTEP,
):
    """Draw one ``(timestep, label)`` pair uniformly over ``intervals``.

    Falls back to a uniform timestep with label ``0`` (which masks the sample
    out of the loss) when the sample has no labelled interval left.
    """
    rng = rng or random
    total = sum(interval.size for interval in intervals)
    if total <= 0:
        return rng.randrange(max_timestep), 0

    offset = rng.randrange(total)
    for interval in intervals:
        if offset < interval.size:
            return interval.lower + offset, interval.label
        offset -= interval.size

    # Unreachable: offset < total is always consumed by the loop above.
    last = intervals[-1]
    return last.upper - 1, last.label


def sample_timesteps_for_batch(
    batch_labels: Sequence[Dict[str, int]],
    rng: Optional[random.Random] = None,
    max_timestep: int = MAX_TIMESTEP,
):
    """Sample one ``(timestep, label)`` per sample of a batch."""
    timesteps: List[int] = []
    labels: List[int] = []
    for labels_of_sample in batch_labels:
        intervals = extract_valid_intervals(labels_of_sample or {}, max_timestep)
        timestep, label = sample_timestep(intervals, rng, max_timestep)
        timesteps.append(timestep)
        labels.append(label)
    return timesteps, labels


def load_confidences(directory: str) -> Dict[str, Dict[str, float]]:
    """Merge the ``confidence_rank*.jsonl`` shards written by ``compute_confidence.py``.

    Distributed evaluation may pad the last batch with repeated pairs; they
    carry identical values, so later lines simply overwrite earlier ones.
    """
    shards = sorted(f for f in os.listdir(directory) if f.startswith("confidence_rank") and f.endswith(".jsonl"))
    if not shards:
        raise FileNotFoundError(f"No confidence_rank*.jsonl shards in {directory}")
    confidences: Dict[str, Dict[str, float]] = {}
    for shard in shards:
        with open(os.path.join(directory, shard), "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    confidences[str(row.pop("idx"))] = row
    return confidences


def load_pseudo_labels(path: str) -> PseudoLabelDict:
    """Read a pseudo-label json file, keyed by dataset index as a string."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(idx): labels for idx, labels in raw.items()}


SELECTION_MODES = ("top_k", "threshold", "percentile")


def select_pseudo_labels(
    confidences: Dict[str, Dict[str, float]],
    mode: str,
    values: Sequence[float],
    past_labels: Optional[PseudoLabelDict] = None,
    clean_ids: Optional[Iterable[str]] = None,
):
    """Turn per-timestep confidences into a pseudo-label dict.

    For every anchor, pairs are ranked by ``|confidence|`` and the selected
    ones get ``sign(confidence)``; all others get ``0``.  Selection per anchor:

    * ``top_k``: the ``k`` most confident pairs.
    * ``threshold``: pairs with ``|confidence| >= value``.
    * ``percentile``: pairs above the given percentile of ``|confidence|``.

    ``values`` holds one value for all anchors or one per anchor.

    Pairs that were already labelled in a previous iteration (``past_labels``)
    are pushed to the bottom of the ranking for the anchors they were labelled
    at.  Pairs in ``clean_ids`` (the consensus-clean set) keep their human label,
    i.e. ``+1`` at every anchor.  Pairs without any non-zero label are dropped.

    Returns the label dict and the confidence cut-off used for each anchor.
    """
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode {mode!r}; expected one of {SELECTION_MODES}")
    if len(values) not in (1, len(TIMESTEP_ANCHORS)):
        raise ValueError(f"Expected 1 or {len(TIMESTEP_ANCHORS)} selection values, got {len(values)}")
    per_anchor = dict(zip(TIMESTEP_ANCHORS, values if len(values) > 1 else list(values) * len(TIMESTEP_ANCHORS)))
    past_labels = past_labels or {}

    labels: PseudoLabelDict = {idx: {timestep_key(a): 0 for a in TIMESTEP_ANCHORS} for idx in confidences}
    cutoffs: Dict[str, float] = {}
    for anchor in TIMESTEP_ANCHORS:
        key = timestep_key(anchor)
        ranked = []
        for idx, row in confidences.items():
            value = row.get(key, 0.0)
            if value == 0:
                continue
            magnitude = 0.0 if key in past_labels.get(idx, {}) else abs(value)
            ranked.append((magnitude, idx, 1 if value > 0 else -1))
        ranked.sort(key=lambda item: item[0], reverse=True)

        target = per_anchor[anchor]
        if mode == "top_k":
            chosen = ranked[: int(target)]
        else:
            if mode == "percentile":
                target = float(np.percentile([m for m, _, _ in ranked], target)) if ranked else 0.0
            chosen = [item for item in ranked if item[0] >= target]
        cutoffs[key] = chosen[-1][0] if chosen else float("nan")
        for _, idx, sign in chosen:
            labels[idx][key] = sign

    for idx in map(str, clean_ids or ()):
        labels[idx] = {timestep_key(a): 1 for a in TIMESTEP_ANCHORS}

    labels = {idx: row for idx, row in labels.items() if any(row.values())}
    return labels, cutoffs


def label_histogram(labels: PseudoLabelDict) -> Dict[str, Dict[int, int]]:
    """Count ``-1 / 0 / +1`` labels per anchor, for logging and reports."""
    histogram = {timestep_key(a): {-1: 0, 0: 0, 1: 0} for a in TIMESTEP_ANCHORS}
    for per_sample in labels.values():
        for key, counts in histogram.items():
            counts[int(per_sample.get(key, 0))] += 1
    return histogram
