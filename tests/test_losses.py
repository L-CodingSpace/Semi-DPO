import pytest
import torch
import torch.nn.functional as F

from semi_dpo.losses import LOSS_TYPES, dpo_loss, reward_consensus_weight


def losses(seed=0, pairs=6):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2 * pairs, generator=g), torch.rand(2 * pairs, generator=g)


def reference_dpo(model, ref, beta):
    model_w, model_l = model.chunk(2)
    ref_w, ref_l = ref.chunk(2)
    return -F.logsigmoid(beta * ((ref_w - ref_l) - (model_w - model_l))).mean()


def test_sigmoid_matches_diffusion_dpo():
    model, ref = losses()
    out = dpo_loss(model, ref, beta=10.0)
    assert torch.allclose(out.loss, reference_dpo(model, ref, 10.0))
    assert out.num_labelled == 6


def test_all_positive_labels_equal_no_labels():
    model, ref = losses()
    plain = dpo_loss(model, ref, beta=5.0)
    labelled = dpo_loss(model, ref, beta=5.0, labels=torch.ones(6))
    assert torch.allclose(plain.loss, labelled.loss)


def test_negative_label_swaps_winner_and_loser():
    model, ref = losses()
    swapped_model = torch.cat(model.chunk(2)[::-1])
    swapped_ref = torch.cat(ref.chunk(2)[::-1])
    flipped = dpo_loss(model, ref, beta=5.0, labels=-torch.ones(6))
    assert torch.allclose(flipped.loss, reference_dpo(swapped_model, swapped_ref, 5.0))


def test_zero_label_is_masked_out():
    model, ref = losses()
    labels = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, 1.0])
    out = dpo_loss(model, ref, beta=5.0, labels=labels)
    keep = labels != 0
    model_w, model_l = model.chunk(2)
    ref_w, ref_l = ref.chunk(2)
    expected = dpo_loss(
        torch.cat([model_w[keep], model_l[keep]]),
        torch.cat([ref_w[keep], ref_l[keep]]),
        beta=5.0,
        labels=labels[keep],
    )
    assert torch.allclose(out.loss, expected.loss)
    assert out.num_labelled == 4


def test_fully_masked_batch_has_zero_loss_and_gradient():
    model, ref = losses()
    model.requires_grad_(True)
    out = dpo_loss(model, ref, beta=5.0, labels=torch.zeros(6))
    out.loss.backward()
    assert out.loss.item() == 0.0
    assert torch.count_nonzero(model.grad) == 0


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_every_loss_type_is_finite(loss_type):
    model, ref = losses()
    out = dpo_loss(model, ref, beta=5.0, loss_type=loss_type, robust_weight=torch.full((6,), 0.5))
    assert torch.isfinite(out.loss)


def test_gce_with_q_zero_falls_back_to_cross_entropy():
    model, ref = losses()
    gce = dpo_loss(model, ref, beta=5.0, loss_type="gce", gce_q=0.0)
    assert torch.allclose(gce.loss, reference_dpo(model, ref, 5.0))


def test_reward_consensus_weight():
    scores = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    assert torch.allclose(reward_consensus_weight(scores), torch.tensor([0.0, 0.5]))
