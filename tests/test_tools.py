import json

from datasets import Dataset, load_from_disk

from semi_dpo import report, split_consensus

REWARDS = ["clip_score", "pick_score"]


def scored_pairs(path):
    # human winner is image 0 for even rows, image 1 for odd rows
    rows = {
        "idx": list(range(6)),
        "human_scores": [[1.0, 0.0], [0.0, 1.0]] * 3,
        # 0, 1: both models agree with the human label
        # 2: the models disagree with each other
        # 3, 5: the models agree with each other but not with the human label
        # 4: one model ties, the other prefers the human winner
        "clip_score": [[2, 1], [1, 2], [2, 1], [2, 1], [1, 1], [3, 1]],
        "pick_score": [[5, 4], [4, 5], [4, 5], [5, 4], [5, 4], [6, 4]],
    }
    Dataset.from_dict(rows).save_to_disk(str(path))
    return str(path)


def run_split(tmp_path, criterion):
    out = tmp_path / criterion
    args = split_consensus.parse_args(
        [
            "--dataset_name",
            scored_pairs(tmp_path / f"pairs_{criterion}"),
            "--output_dir",
            str(out),
            "--reward_models",
            *REWARDS,
            "--criterion",
            criterion,
            "--test_size",
            "1",
            "--num_proc",
            "1",
        ]
    )
    split_consensus.main(args)
    clean = load_from_disk(str(out / "clean"))
    ids = sorted(list(clean["train"]["idx"]) + list(clean["test"]["idx"]))
    return ids, json.loads((out / "clean_idx.json").read_text()), json.loads((out / "stats.json").read_text())


def test_consensus_requires_agreement_with_human_label(tmp_path):
    ids, clean_idx, stats = run_split(tmp_path, "human_agree")
    assert ids == [0, 1]
    assert len(clean_idx) == 1 and set(clean_idx) < {0, 1}
    assert stats["agreement_with_human"] == {"clip_score": 0.5, "pick_score": 0.5}


def test_consensus_models_agree_ignores_human_label(tmp_path):
    ids, _, _ = run_split(tmp_path, "models_agree")
    assert ids == [0, 1, 3, 4, 5]


def write_run(directory, scores):
    directory.mkdir()
    for i, value in enumerate(scores):
        (directory / f"{i}-0.json").write_text(json.dumps({"prompt": "p", "pick_score": value, "hps_score": 0.3}))


def test_report_win_rate(tmp_path, capsys):
    write_run(tmp_path / "base", [1.0, 1.0, 1.0, 1.0])
    write_run(tmp_path / "model", [2.0, 2.0, 0.0, 2.0])
    output = tmp_path / "table.csv"
    report.main(
        report.parse_args([str(tmp_path / "model"), "--baseline", str(tmp_path / "base"), "--output", str(output)])
    )
    row = output.read_text().splitlines()[1]
    assert "0.75" in row  # pick_score win rate
    assert "75.0%" in capsys.readouterr().out
