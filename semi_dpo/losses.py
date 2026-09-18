"""Diffusion-DPO objectives with optional timestep-conditional pseudo-labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

LOSS_TYPES = ("sigmoid", "hinge", "ipo", "gce", "sce")


@dataclass
class DPOOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    implicit_acc: torch.Tensor
    num_labelled: torch.Tensor


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error reduced over every dimension except the batch."""
    loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    return loss.mean(dim=list(range(1, loss.ndim)))


def dpo_loss(
    model_losses: torch.Tensor,
    ref_losses: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
    labels: Optional[torch.Tensor] = None,
    robust_weight: Optional[torch.Tensor] = None,
    gce_q: float = 0.5,
    sce_beta: float = 0.5,
    log_zero: float = -6.0,
) -> DPOOutput:
    """Diffusion-DPO loss on a batch laid out as ``[winners; losers]``.

    Args:
        model_losses: per-sample denoising loss of the trained model, shape ``(2B,)``.
        ref_losses: same for the frozen reference model.
        beta: KL regularisation strength.
        loss_type: one of :data:`LOSS_TYPES`.
        labels: optional pseudo-labels in ``{-1, 0, +1}`` of shape ``(B,)``.
            ``+1`` keeps the human preference, ``-1`` swaps winner and loser
            and ``0`` removes the pair from the loss.  ``None`` means all ``+1``,
            which reduces to standard Diffusion-DPO.
        robust_weight: per-pair mixing weight for the ``gce``/``sce`` losses.
    """
    model_w, model_l = model_losses.chunk(2)
    ref_w, ref_l = ref_losses.chunk(2)
    if labels is None:
        labels = torch.ones_like(model_w)
    labels = labels.to(model_w.dtype)

    # Multiplying by the label flips the preference direction for label -1.
    model_diff = (model_w - model_l) * labels
    ref_diff = (ref_w - ref_l) * labels
    logits = beta * (ref_diff - model_diff)

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(logits)
    elif loss_type == "hinge":
        losses = torch.relu(1.0 - logits)
    elif loss_type == "ipo":
        # (log_ratio - 1 / (2 beta))^2 expressed with logits = beta * log_ratio.
        losses = (logits - 0.5).pow(2) / beta**2
    elif loss_type == "gce":
        # Generalized Cross Entropy, https://arxiv.org/abs/1805.07836
        weight = robust_weight if robust_weight is not None else torch.ones_like(logits)
        ce = -F.logsigmoid(logits)
        if gce_q > 0:
            gce = (1 - torch.sigmoid(logits).pow(gce_q)) / gce_q
        else:
            gce = ce
        losses = (1 - weight) * ce + weight * gce
    elif loss_type == "sce":
        # Symmetric Cross Entropy, https://arxiv.org/abs/1908.06112
        ce = -F.logsigmoid(logits)
        rce = -torch.sigmoid(-logits) * log_zero
        losses = ce + sce_beta * rce
    else:
        raise ValueError(f"Unknown loss type {loss_type!r}; expected one of {LOSS_TYPES}")

    mask = labels != 0
    num_labelled = mask.sum()
    loss = (losses * mask).sum() / num_labelled.clamp(min=1)

    # Implicit accuracy over labelled pairs, counting ties as half correct.
    correct = (logits > 0).float() + 0.5 * (logits == 0).float()
    implicit_acc = (correct * mask).sum() / num_labelled.clamp(min=1)

    return DPOOutput(loss=loss, logits=logits, implicit_acc=implicit_acc, num_labelled=num_labelled)


def reward_consensus_weight(reward_scores: torch.Tensor) -> torch.Tensor:
    """Fraction of reward models that *disagree* with the human winner.

    Args:
        reward_scores: ``(B, num_models, 2)`` scores for ``[winner, loser]``.
    """
    agree = (reward_scores[:, :, 0] > reward_scores[:, :, 1]).float().mean(dim=1)
    return 1 - agree
