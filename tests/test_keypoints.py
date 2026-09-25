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
