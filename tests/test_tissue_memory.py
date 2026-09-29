from types import SimpleNamespace

import numpy as np

from surgscene.tissue_memory import TissueMemory, distances, plane_depth


def test_memory_keeps_tissue_under_an_occluder_and_adapts_to_change():
    mem = TissueMemory((4, 4), K=5)
    d = np.full((4, 4), 20.0, np.float32)
    for _ in range(5):
        mem.update(d, np.ones((4, 4), bool))
    occ = np.ones((4, 4), bool)
    occ[1, 1] = False  # instrument over (1, 1): not written
    for _ in range(10):
        mem.update(np.full((4, 4), 35.0, np.float32), occ)
    m = mem.disparity(0, 4, 0, 4)
    assert m[1, 1] == 20.0  # remembered from before the occlusion
    assert m[0, 0] == 35.0  # the rest followed the (moved) tissue within K observations
    assert np.isnan(TissueMemory((2, 2)).disparity(0, 2, 0, 2)).all()  # never seen


def test_plane_depth_and_signed_gap():
    rect = SimpleNamespace(f=100.0, B=5.0, cx=10.0, cy=10.0)
    plane = (np.array([0.0, 0.0, 1.0]), 200.0)  # Z = 200 everywhere
    ys, xs = np.mgrid[0:21, 0:21].astype(float)
    Z = plane_depth(xs, ys, plane, rect)
    assert np.allclose(Z, 200.0)
    gap, near = distances(np.array([0.0, 0.0, 190.0]), Z, 0, 0, rect)
    # tip 10 mm in front of the surface; "near" is a robust low percentile, so slightly above the minimum
    assert abs(gap - 10.0) < 1e-9 and 10.0 <= near < 11.0
