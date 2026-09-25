import numpy as np
import pytest

from surgscene.evaluation import ci, cluster_bootstrap, frame_stats, stack, summarize


def square(h=64, w=64, y=16, x=16, s=20, c=1):
    m = np.zeros((h, w), np.uint8)
    m[y:y + s, x:x + s] = c
    return m


def test_perfect_prediction():
    gt = square()
    S = stack([frame_stats(gt, gt, 2)])
    r = summarize(S)
    assert r["dice"][1] == 1 and r["iou"][1] == 1 and r["bf"][1] == 1 and r["hd95"][1] == 0


def test_shifted_square_dice_and_hd():
    gt, pred = square(x=16), square(x=20)  # 20x20 square shifted 4 px
    r = summarize(stack([frame_stats(pred, gt, 2, tol=1)]))
    assert r["dice"][1] == pytest.approx(2 * 16 * 20 / 800)
    assert r["hd95"][1] == pytest.approx(4, abs=0.5)
    r_loose = summarize(stack([frame_stats(pred, gt, 2, tol=5)]))
    assert r_loose["bf"][1] == 1 and r["bf"][1] < 1


def test_ignore_index_excluded():
    gt = square()
    gt[:, :8] = 255
    pred = gt.copy()
    pred[:, :8] = 1  # wrong, but ignored
    assert summarize(stack([frame_stats(pred, gt, 2)]))["dice"][1] == 1


def test_pooling_is_pixel_weighted():
    big, small = square(s=30), square(s=4)
    S = stack([frame_stats(big, big, 2), frame_stats(np.zeros_like(small), small, 2)])
    r = summarize(S)
    assert r["dice"][1] == pytest.approx(2 * 900 / (2 * 900 + 16))


def test_ece_perfectly_calibrated_is_zero():
    rng = np.random.default_rng(0)
    gt = np.zeros((200, 200), np.uint8)
    conf = np.full(gt.shape, 0.7)
    pred = (rng.random(gt.shape) > 0.7).astype(np.uint8)  # correct w.p. 0.7
    r = summarize(stack([frame_stats(pred, gt, 2, conf=conf, classes=[])]))
    assert r["ece"] == pytest.approx(0, abs=0.01)


def test_bootstrap_resamples_groups_not_frames():
    gt = square()
    good, bad = frame_stats(gt, gt, 2), frame_stats(np.zeros_like(gt), gt, 2)
    # group A: 10 perfect frames, group B: 1 empty prediction. Frame-level resampling would give
    # dice ~1 almost always; group-level resampling must often drop or double group B.
    S = stack([good] * 10 + [bad])
    groups = ["A"] * 10 + ["B"]
    d = cluster_bootstrap(S, groups, lambda r: r["dice"][1], n_boot=500)
    lo, hi = ci(d)
    assert lo < 0.95 and hi == 1.0
    assert (d == 1.0).mean() > 0.2  # replicates that drew only group A
