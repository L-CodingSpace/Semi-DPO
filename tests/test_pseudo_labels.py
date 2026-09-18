import json
import random

import pytest

from semi_dpo import pseudo_labels as pl


def all_anchors(value):
    return {pl.timestep_key(a): value for a in pl.TIMESTEP_ANCHORS}


def test_label_from_confidence():
    assert pl.label_from_confidence(2.0, 1.0) == 1
    assert pl.label_from_confidence(-2.0, 1.0) == -1
    assert pl.label_from_confidence(0.5, 1.0) == 0
    assert pl.label_from_confidence(1.0, 1.0) == 0


def test_intervals_are_disjoint_and_skip_unlabelled():
    labels = {"timestep_50": 1, "timestep_150": 0, "timestep_250": -1, "timestep_350": 1}
    intervals = pl.extract_valid_intervals(labels)
    assert [(i.anchor, i.label) for i in intervals] == [(50, 1), (250, -1), (350, 1)]
    assert [(i.lower, i.upper) for i in intervals] == [(0, 100), (200, 300), (300, 400)]


def test_every_anchor_labelled_tiles_whole_schedule():
    intervals = pl.extract_valid_intervals(all_anchors(1))
    assert intervals[0].lower == 0 and intervals[-1].upper == pl.MAX_TIMESTEP
    assert sum(i.size for i in intervals) == pl.MAX_TIMESTEP
    for a, b in zip(intervals, intervals[1:]):
        assert a.upper == b.lower


def test_close_custom_anchors_are_clipped():
    intervals = pl.extract_valid_intervals({"timestep_100": 1, "timestep_120": -1})
    assert [(i.lower, i.upper, i.label) for i in intervals] == [(50, 150, 1), (150, 170, -1)]


def test_sampled_timestep_matches_its_interval_label():
    labels = {"timestep_50": 1, "timestep_550": -1, "timestep_950": 1}
    intervals = pl.extract_valid_intervals(labels)
    rng = random.Random(0)
    seen = set()
    for _ in range(2000):
        t, label = pl.sample_timestep(intervals, rng)
        owner = [i for i in intervals if i.lower <= t < i.upper]
        assert len(owner) == 1 and owner[0].label == label
        seen.add(owner[0].anchor)
    assert seen == {50, 550, 950}


def test_unlabelled_sample_is_masked():
    t, label = pl.sample_timestep([], random.Random(0))
    assert 0 <= t < pl.MAX_TIMESTEP and label == 0
    timesteps, labels = pl.sample_timesteps_for_batch([{}, all_anchors(0)], random.Random(0))
    assert labels == [0, 0]


def confidences():
    # idx -> confidence at every anchor
    return {
        "0": all_anchors(5.0),
        "1": all_anchors(-4.0),
        "2": all_anchors(3.0),
        "3": all_anchors(0.5),
        "4": all_anchors(0.0),
    }


def test_select_top_k():
    labels, cutoffs = pl.select_pseudo_labels(confidences(), "top_k", [2])
    assert set(labels) == {"0", "1"}
    assert labels["0"]["timestep_50"] == 1 and labels["1"]["timestep_50"] == -1
    assert cutoffs["timestep_50"] == 4.0


def test_select_threshold_per_anchor():
    values = [3.0] * 5 + [100.0] * 5
    labels, _ = pl.select_pseudo_labels(confidences(), "threshold", values)
    assert set(labels) == {"0", "1", "2"}
    assert labels["2"]["timestep_450"] == 1 and labels["2"]["timestep_550"] == 0


def test_select_percentile():
    labels, _ = pl.select_pseudo_labels(confidences(), "percentile", [50])
    # |z| of labelled candidates: 5, 4, 3, 0.5 -> median 3.5
    assert set(labels) == {"0", "1"}


def test_clean_pairs_forced_positive_and_past_pairs_ranked_last():
    past = {"0": all_anchors(1)}
    labels, _ = pl.select_pseudo_labels(confidences(), "top_k", [2], past_labels=past, clean_ids=["3", "99"])
    assert set(labels) == {"1", "2", "3", "99"}
    assert labels["3"] == all_anchors(1) and labels["99"] == all_anchors(1)


def test_invalid_selection_values():
    with pytest.raises(ValueError):
        pl.select_pseudo_labels(confidences(), "top_k", [1, 2])
    with pytest.raises(ValueError):
        pl.select_pseudo_labels(confidences(), "median", [1])


def test_load_confidences_merges_shards(tmp_path):
    rows = [{"idx": 1, "timestep_50": 1.0}, {"idx": 2, "timestep_50": -1.0}]
    (tmp_path / "confidence_rank000.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "confidence_rank001.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert pl.load_confidences(str(tmp_path)) == {"1": {"timestep_50": 1.0}, "2": {"timestep_50": -1.0}}
