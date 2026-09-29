import numpy as np
import pytest
import torch

from surgscene.keypoints import decode, focal_loss, render_heatmaps


def test_decode_recovers_subpixel_location():
    kp = np.array([[10.3, 20.7], [50.0, 5.25], [3.9, 40.1]])
    hm = torch.from_numpy(render_heatmaps(kp, 64, 64))[None]
    out, conf = decode(hm)
    # log of a Gaussian is exactly quadratic, so the refinement should be near exact
    np.testing.assert_allclose(out[0], kp, atol=0.02)
    assert conf.min() > 0.9


def test_missing_keypoint_renders_empty():
    hm = render_heatmaps(np.array([[np.nan, np.nan], [100.0, 100.0]]), 32, 32)
    assert hm.max() == 0


def test_focal_loss_prefers_correct_heatmap():
    kp = np.array([[16.0, 16.0]])
    target = torch.from_numpy(render_heatmaps(kp, 32, 32))[None]
    good = torch.logit(target.clamp(1e-3, 1 - 1e-3))
    bad = torch.logit(torch.from_numpy(render_heatmaps(np.array([[5.0, 5.0]]), 32, 32))[None].clamp(1e-3, 1 - 1e-3))
    assert focal_loss(good, target) < focal_loss(bad, target)


def test_focal_argmax_positive_exists_for_subpixel_keypoints():
    # keypoint at a pixel corner: no pixel reaches 0.99, so the threshold rule has no positive
    target = torch.from_numpy(render_heatmaps(np.array([[16.5, 16.5], [np.nan, np.nan]]), 32, 32))[None]
    assert target[0, 0].max() < 0.99
    logits = torch.full_like(target, -4.6)
    # a higher peak logit must lower the loss under "argmax"; under "threshold" the peak is only a negative
    hi = logits.clone()
    hi[0, 0, 16, 16] = 4.0
    assert focal_loss(hi, target, pos_mode="argmax") < focal_loss(logits, target, pos_mode="argmax")
    assert focal_loss(hi, target, pos_mode="threshold") >= focal_loss(logits, target, pos_mode="threshold")
    # the empty (unlabeled) channel contributes no positive
    assert focal_loss(logits, target, pos_mode="argmax").isfinite()


def test_nll_loss_gradient_finite_with_unlabeled_keypoints():
    from surgscene.kp_vit import nll_loss
    hm = torch.rand(2, 3, 16, 16)
    lv = torch.zeros(2, 3, 16, 16, requires_grad=True)
    kp = torch.full((2, 3, 2), 5.0)
    kp[0, 1] = float("nan")
    nll_loss(lv, hm, kp).backward()
    assert torch.isfinite(lv.grad).all()
