"""End-to-end run of the Semi-DPO pipeline on tiny random models (CPU, ~1 min).

Needs network access to download the ``hf-internal-testing`` tiny pipelines
once.  Run with ``pytest tests/test_smoke.py``.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from datasets import Dataset, Image, Sequence
from PIL import Image as PILImage

TINY_MODELS = {
    "sd15": "hf-internal-testing/tiny-stable-diffusion-pipe",
    "sdxl": "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_model(family):
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(TINY_MODELS[family])
    except Exception as error:  # offline
        pytest.skip(f"cannot download {TINY_MODELS[family]}: {error}")


def make_dataset(path, num_pairs=8):
    rng = np.random.default_rng(0)

    def image():
        return PILImage.fromarray(rng.integers(0, 255, (80, 72, 3), dtype=np.uint8))

    rows = {
        "idx": list(range(100, 100 + num_pairs)),
        "prompt": [f"a photo of object {i}" for i in range(num_pairs)],
        "images": [[image(), image()] for _ in range(num_pairs)],
        "human_scores": [[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(num_pairs)],
    }
    for name in ("clip_score", "aesthetic_score", "image_reward_score", "pick_score", "hps_score"):
        rows[name] = rng.random((num_pairs, 2)).tolist()
    Dataset.from_dict(rows).cast_column("images", Sequence(Image())).save_to_disk(path)
    return path


def cli(**options):
    """``cli(output_dir="x", seed=0)`` -> ``["--output_dir", "x", "--seed", "0"]``."""
    args = []
    for key, value in options.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        args += [f"--{key}", *map(str, values)]
    return args


def run(module, *args):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, WANDB_MODE="disabled", HF_HUB_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-m", f"semi_dpo.{module}", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module} failed:\n{result.stdout[-3000:]}\n{result.stderr[-6000:]}"
    return result.stdout


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_full_pipeline(family, tmp_path):
    model = tiny_model(family)
    data = make_dataset(tmp_path / "pairs")
    common = cli(model_family=family, pretrained_model_name_or_path=model, resolution=64)
    optim = cli(train_batch_size=2, learning_rate=1e-5, seed=0)

    # Stage 1: Diffusion-DPO on "clean" pairs.
    stage1 = tmp_path / "stage1"
    run(
        "train",
        *common,
        *optim,
        *cli(train_dataset_name=data, output_dir=stage1, num_val_samples=2, max_train_steps=2, checkpointing_steps=2),
    )
    checkpoint = stage1 / "checkpoint-2"
    assert (checkpoint / "unet" / "config.json").exists()
    assert (stage1 / "final" / "unet").exists()

    # Stage 2: per-timestep confidences, statistics and pseudo-labels.
    confidence_dir = tmp_path / "confidence"
    run(
        "compute_confidence",
        *common,
        *cli(checkpoint=checkpoint, dataset_name=data, output_dir=confidence_dir, batch_size=4),
    )
    rows = [json.loads(line) for line in (confidence_dir / "confidence_rank000.jsonl").read_text().splitlines()]
    assert sorted(r["idx"] for r in rows) == list(range(100, 108))
    assert all(len(r) == 11 for r in rows)

    run("summarize_confidence", *cli(confidence_dir=confidence_dir))
    assert (confidence_dir / "summary.csv").exists()

    clean_ids = tmp_path / "clean_idx.json"
    clean_ids.write_text(json.dumps([100, 101]))
    labels_path = tmp_path / "pseudo_labels.json"
    run(
        "build_pseudo_labels",
        *cli(confidence_dir=confidence_dir, output=labels_path, mode="top_k", values=3, clean_idx_path=clean_ids),
    )
    labels = json.loads(labels_path.read_text())
    assert labels["100"] == labels["101"] == {f"timestep_{t}": 1 for t in range(50, 1000, 100)}

    # Stage 3: retrain on clean + pseudo-labelled pairs, starting from stage 1
    # and (as in the SD 1.5 runs) using it as the DPO reference.
    stage3 = tmp_path / "stage3"
    run(
        "train",
        *common,
        *optim,
        *cli(train_dataset_name=data, pseudo_label_path=labels_path, output_dir=stage3, num_val_samples=1),
        *cli(pretrained_unet_name_or_path=checkpoint, ref_model_name_or_path=checkpoint / "unet"),
        *cli(max_train_steps=2, checkpointing_steps=1),
    )
    assert (stage3 / "checkpoint-1" / "unet").exists() and (stage3 / "checkpoint-2" / "unet").exists()
