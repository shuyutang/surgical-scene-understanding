import numpy as np

from surgscene.sam2_track import CONF_MIN, load_masks, prompt_points, save_masks, start_frame


def test_prompt_points_extends_along_shaft_inside_image():
    kp = np.zeros((10, 2))
    kp[0], kp[1] = [500, 400], [450, 400]  # PSM1 shaft, wrist: shaft direction +x
    kp[2:5] = [[420, 400], [400, 390], [400, 410]]
    kp[5], kp[6] = [100, 100], [150, 100]  # PSM3: shaft direction -x, extension leaves the image
    kp[7:10] = [[180, 100], [200, 90], [200, 110]]
    conf = np.full(10, 0.9)
    conf[4] = CONF_MIN - 0.1
    pts, labels = prompt_points(kp, conf, (600, 800))
    assert len(pts[0]) == 4 + 3 and pts[0][-1] == [650.0, 400.0]
    assert len(pts[1]) == 5 + 2  # x = 50 and x = 0 are in the image, x = -50 is not
    assert labels == [[1] * len(pts[0]), [1] * len(pts[1])]


def test_start_frame_prefers_shaft_and_wrist():
    conf = np.zeros((5, 10))
    conf[1, [2, 3, 4, 7, 8, 9]] = 0.9  # 3 keypoints each, no shaft/wrist
    conf[3] = 0.9
    assert start_frame(conf) == 3
    conf[3] = 0.0
    assert start_frame(conf) == 1
    assert start_frame(np.zeros((5, 10))) is None


def test_mask_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.random((3, 2, 7, 13)) > 0.5
    save_masks(tmp_path / "m.npz", m, start=2)
    d = load_masks(tmp_path / "m.npz")
    assert (d["masks"] == m).all() and int(d["start"]) == 2
