# Semi-DPO: Learning from Noisy Preferences

Official code for **Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization** (ICLR 2026).

Xinxin Liu, Ming Li, Zonglin Lyu, Yuzhang Shang, Chen Chen — University of Central Florida

[Paper (arXiv)](https://arxiv.org/abs/2604.24952) · [OpenReview](https://openreview.net/forum?id=rRc04jyoAk) · [Project page](https://liming-ai.github.io/SemiDPO/)

<!-- TODO: add links to the released checkpoints once they are public. -->

Human preferences over images are multi-dimensional (composition, aesthetics, detail, text alignment), but
preference datasets such as Pick-a-Pic record a single winner per pair. When the two images win on different
dimensions, the holistic label sends conflicting gradients to Diffusion-DPO. Semi-DPO treats this as learning with
noisy labels:

1. **Multi-reward consensus.** A pair is *clean* only if every reward model (CLIP, LAION aesthetics, ImageReward,
   PickScore, HPSv2) prefers the human winner. This keeps about 21% of Pick-a-Pic v2. All other pairs are treated as
   unlabelled.
2. **Iterative self-training with timestep-conditional pseudo-labels.** A model trained on the clean pairs acts as
   an implicit classifier. For every pair and every timestep interval, the sign of its DPO logit becomes a
   pseudo-label and the magnitude its confidence. Confident labels are kept per interval, and the next model is
   trained on the clean pairs plus the pseudo-labelled pairs. The loop then repeats.

## Results

SD 1.5 trained on Pick-a-Pic v2, evaluated on the Pick-a-Pic v2 test prompts (Table 1 of the paper):

| Method | ImageReward | HPSv2.1 | PickScore | Aesthetic | CLIP | MPS |
|---|---|---|---|---|---|---|
| SD 1.5 | 0.085 | 0.250 | 20.566 | 5.421 | 0.273 | 9.635 |
| Diffusion-DPO | 0.297 | 0.261 | 20.948 | 5.549 | 0.279 | 10.144 |
| Diffusion-KTO | 0.629 | 0.281 | 21.064 | 5.659 | 0.281 | 10.226 |
| **Semi-DPO** | **0.801** | **0.288** | **21.524** | **5.801** | **0.281** | **11.030** |

Win rate of Semi-DPO on the same prompts (Table 2):

| Semi-DPO vs. | ImageReward | HPSv2.1 | PickScore | Aesthetic | CLIP | MPS |
|---|---|---|---|---|---|---|
| Diffusion-DPO | 72.9% | 82.1% | 76.3% | 72.6% | 53.5% | 69.0% |
| Diffusion-KTO | 60.2% | 59.5% | 73.0% | 64.5% | 50.4% | 70.2% |

The paper also reports HPS v2 and Parti-Prompts results, GenEval (overall 47.31 vs. 43.00 for Diffusion-DPO), and
SDXL experiments. MPS and GenEval are computed with their official repositories and are not part of this code.

## Installation

```bash
conda create -n semi-dpo python=3.10 -y
conda activate semi-dpo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version
pip install -r requirements.txt
pip install -e .
```

The reward models (`clip`, `image-reward`, `hpsv2`) are only needed for scoring data, validation during training
and evaluation. Weights are cached in `~/.cache/semi_dpo` (override with `SEMI_DPO_CACHE`).

## Data format

Every tool works on a 🤗 `datasets` folder (or hub dataset) of preference pairs:

| column | content |
|---|---|
| `idx` | stable integer id of the pair (keys the pseudo-labels) |
| `prompt` | text prompt |
| `images` | `[image_0, image_1]` |
| `human_scores` | `[label_0, label_1]` (or `winner_idx`) |
| `clip_score`, `aesthetic_score`, `image_reward_score`, `pick_score`, `hps_score` | `[score_0, score_1]`, added by `score_pairs.py` |

`semi_dpo/prepare_pickapic.py` builds this format from
[Pick-a-Pic v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2). It drops ties, which leaves the 851,293
training pairs also used by Diffusion-DPO.

## Pipeline

Each stage has a launch script in `scripts/`. `MODEL=sd15` or `MODEL=sdxl` selects the model family. The distributed
setup comes from `NUM_MACHINES`, `NUM_PROCESSES` (total GPUs), `MACHINE_RANK` and `MAIN_PROCESS_IP` (see
`scripts/common.sh`).

```bash
# Stage 0: build, score and split the dataset -> data/consensus/{clean, clean_idx.json, stats.json}
bash scripts/0_prepare_data.sh

# Iteration 0: Diffusion-DPO on the clean pairs
MODEL=sd15 bash scripts/1_train_clean.sh

# Iteration 1: pseudo-label with iteration 0, then retrain from it
MODEL=sd15 ITER=1 CHECKPOINT=runs/sd15/iter0/checkpoint-1600 bash scripts/2_pseudo_label.sh
MODEL=sd15 ITER=1 INIT_CHECKPOINT=runs/sd15/iter0/checkpoint-1600 bash scripts/3_train_pseudo.sh

# Iteration 2: same again; earlier pseudo-labels are passed so they are ranked last
MODEL=sd15 ITER=2 CHECKPOINT=runs/sd15/iter1/checkpoint-4000 \
  PAST_LABELS=runs/sd15/iter1/pseudo_labels/pseudo_labels.json bash scripts/2_pseudo_label.sh
MODEL=sd15 ITER=2 INIT_CHECKPOINT=runs/sd15/iter1/checkpoint-4000 bash scripts/3_train_pseudo.sh

# Evaluation on Pick-a-Pic v2, HPS v2 and Parti-Prompts
MODEL=sd15 NAME=semi_dpo CHECKPOINT=runs/sd15/iter2/checkpoint-4000 bash scripts/4_evaluate.sh
python semi_dpo/report.py eval/semi_dpo/pickapic --baseline eval/sd15_base/pickapic
```

What each stage does:

| Stage | Tool | Output |
|---|---|---|
| Prepare | `prepare_pickapic.py` | pair dataset without ties |
| Score | `score_pairs.py` | reward scores for both images of every pair (resumable) |
| Split | `split_consensus.py` | clean `train`/`test` pairs, `clean_idx.json`, agreement statistics |
| Train | `train.py` | Diffusion-DPO; with `--pseudo_label_path` it samples each pair's timestep from its labelled intervals and flips or masks the preference accordingly |
| Confidence | `compute_confidence.py` | `beta * (ref_diff - model_diff)` for every pair at t = 50, 150, …, 950 |
| Accuracy | `summarize_confidence.py` | per-timestep accuracy and confidence percentiles on the clean test pairs (Table 7) |
| Select | `build_pseudo_labels.py` | `{idx: {"timestep_50": ±1/0, …}}`: clean pairs get `+1` everywhere, noisy pairs keep confident labels |
| Evaluate | `evaluate.py`, `report.py` | generated images, reward scores, means and win rates |

Every tool documents its arguments with `--help`.

### Training details

SD 1.5 (paper, Appendix 6.9): 32 GPUs, 4 pairs per GPU, 4 accumulation steps (global batch 512), β = 2500,
400 warm-up steps. Iteration 0 uses learning rate 4e-9 for 1,600 steps; iterations 1 and 2 use 4e-10 for 4,000 steps.
`--scale_lr` multiplies the learning rate by the global batch size, as in Diffusion-DPO.

Pseudo-label selection: the threshold of each timestep interval starts at the 80th percentile of its confidence.
Intervals where the model is less than 70% accurate on the clean test pairs (t > 650 for SD 1.5) get a stricter
threshold. `scripts/2_pseudo_label.sh` prints the per-timestep accuracy; set `SELECT_MODE` / `SELECT_VALUES`
(`percentile`, `threshold` or `top_k`, one value or ten) accordingly.

Clean anchor pairs: every iteration labels a set of clean pairs `+1` at all timesteps (`CLEAN_IDX`, default: all
clean training pairs). The SD 1.5 runs split the clean training pairs into three disjoint thirds by `idx` and used
the first third in iteration 1 and the second in iteration 2, for example:

```bash
python -c "import json, numpy as np; ids = sorted(json.load(open('data/consensus/clean_idx.json')))
for i, part in enumerate(np.array_split(ids, 3)): json.dump(part.tolist(), open(f'data/consensus/clean_idx_part{i}.json', 'w'))"
MODEL=sd15 ITER=1 CLEAN_IDX=data/consensus/clean_idx_part0.json CHECKPOINT=... bash scripts/2_pseudo_label.sh
```

Reference model: the SD 1.5 runs used the previous iteration's model as the DPO reference, the SDXL runs the base
model. `--ref_model_name_or_path` controls this and the scripts follow the original runs. SDXL is trained with FSDP
(`configs/fsdp.yaml`) and gradient checkpointing.

## Repository layout

```
semi_dpo/
  train.py                  Diffusion-DPO / Semi-DPO training (SD 1.5 and SDXL)
  compute_confidence.py     per-timestep implicit-classifier confidence
  summarize_confidence.py   accuracy and confidence statistics per timestep
  build_pseudo_labels.py    confidence -> pseudo-label file
  prepare_pickapic.py       Pick-a-Pic -> pair dataset
  score_pairs.py            reward-model scores for every pair
  split_consensus.py        multi-reward consensus split
  evaluate.py, report.py    image generation, scoring and comparison
  losses.py                 DPO losses with pseudo-label masking
  pseudo_labels.py          timestep intervals, sampling and selection
  models.py, data.py        model families, checkpoint loading, pair preprocessing
  reward_models.py          CLIP, aesthetic, ImageReward, PickScore, HPSv2
configs/                    accelerate configs (multi-GPU, FSDP)
scripts/                    launch scripts for every stage
tests/                      unit tests and a CPU end-to-end run on tiny models
```

## Tests

```bash
pip install pytest
pytest tests
```

`tests/test_smoke.py` runs the whole pipeline (train → confidence → pseudo-labels → retrain) on the tiny SD 1.5 and
SDXL test models from `hf-internal-testing` on CPU in under a minute.

## Citation

```bibtex
@inproceedings{liu2026semidpo,
  title     = {Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization},
  author    = {Liu, Xinxin and Li, Ming and Lyu, Zonglin and Shang, Yuzhang and Chen, Chen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## Acknowledgements

The training code builds on [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO) and
[diffusers](https://github.com/huggingface/diffusers). Reward models come from
[CLIP](https://github.com/openai/CLIP), [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor),
[ImageReward](https://github.com/THUDM/ImageReward), [PickScore](https://github.com/yuvalkirstain/PickScore) and
[HPSv2](https://github.com/tgxs002/HPSv2).

## License

[Apache License 2.0](LICENSE).
