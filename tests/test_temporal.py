import numpy as np

from surgscene.temporal import KFParams, filter_keypoints, jitter, kalman_track

rng = np.random.default_rng(0)


def trajectory(T=300):
    t = np.arange(T)
    return np.stack([200 + 80 * np.sin(t / 40), 150 + 50 * np.cos(t / 55)], 1)


def test_filter_reduces_noise_and_is_consistent():
    gt = trajectory()
    z = gt + rng.normal(scale=3.0, size=gt.shape)
    tr = kalman_track(z, np.full(len(z), 3.0), np.ones(len(z)), KFParams(q=0.3, r0=1.0))
    err_raw = np.linalg.norm(z - gt, axis=1)[20:].mean()
    err_kf = np.linalg.norm(tr.x_filt[:, :2] - gt, axis=1)[20:].mean()
    assert err_kf < 0.7 * err_raw
    # NIS of a consistent 2D filter averages ~2
    assert 1.5 < np.nanmean(tr.nis[20:]) < 2.6


def test_predict_through_missing_and_covariance_grows():
    gt = trajectory()
    z = gt.copy()
    conf = np.ones(len(z))
    conf[100:130] = 0.0  # occluded: no confident measurement
    pos, std, unknown, _ = filter_keypoints(z[:, None], np.ones((len(z), 1)), conf[:, None], KFParams(q=0.5, unknown_px=10))
    assert np.all(np.diff(std[100:130, 0]) > 0)
    assert unknown[129, 0] and not unknown[99, 0]
    assert np.linalg.norm(pos[105, 0] - gt[105]) < 3.0  # constant-velocity prediction is good short-term


def test_outlier_rejected_then_reinit_on_persistent_jump():
    gt = trajectory()
    z = gt.copy()
    z[150] += 60  # one gross outlier
    tr = kalman_track(z, np.ones(len(z)), np.ones(len(z)), KFParams(q=0.5))
    assert not tr.updated[150] and np.linalg.norm(tr.x_filt[150, :2] - gt[150]) < 3
    z2 = gt.copy()
    z2[200:] += 80  # the target really jumped: re-initialize after max_rejects
    tr2 = kalman_track(z2, np.ones(len(z2)), np.ones(len(z2)), KFParams(q=0.5, max_rejects=3))
    assert np.linalg.norm(tr2.x_filt[210, :2] - z2[210]) < 3


def test_smoother_beats_filter_and_jitter_metric():
    gt = trajectory()
    z = gt + rng.normal(scale=3.0, size=gt.shape)
    args = (z[:, None], np.full((len(z), 1), 3.0), np.ones((len(z), 1)), KFParams(q=0.3))
    f, *_ = filter_keypoints(*args, lag=0)
    s, *_ = filter_keypoints(*args, lag=5)
    e = lambda p: np.linalg.norm(p[20:-20, 0] - gt[20:-20], axis=1).mean()
    assert e(s) < e(f)
    assert np.nanmean(jitter(s, gt[:, None])) < np.nanmean(jitter(z[:, None], gt[:, None]))
    assert np.nanmax(jitter(gt[:, None], gt[:, None])) == 0
