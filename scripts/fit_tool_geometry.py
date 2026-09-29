"""v3 Phase C: instrument geometry in the dVRK tool frame, from train trajectories (0-17) only.

  uv run python scripts/fit_tool_geometry.py

The kinematic tool frame (api_cp_data, per arm) sits at the jaw pivot: tool axis = z, the jaws
open along +-u (close to y). Per train trajectory, a rigid kinematics->camera transform is fitted
from the GT-triangulated jaw pivot (oracle hand-eye, legitimate on training data). Every GT
keypoint is then expressed in the tool frame and the model below is fitted:

  pivot  = 0
  wrist  = (ax, L sin(q5 + b), -L cos(q5 + b))     q5 = wrist-yaw joint (api_jp_data[arm][5])
  tip_a  = c + h u,   tip_b = c - h u              h = jaw half-opening, per frame (from vision)

Writes configs/tool_geometry.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_rectification import RAW, load_gt  # noqa: E402

from surgscene.fusion import ARMS, load_kinematics  # noqa: E402
from surgscene.geometry import load_stereo_ini, register_rigid_robust  # noqa: E402
from surgscene.stage2 import triangulate_seq  # noqa: E402

TRAIN = range(0, 18)


def main():
    loc = {arm: [] for arm in ARMS}
    q5 = {arm: [] for arm in ARMS}
    pivot_res = {arm: [] for arm in ARMS}
    for traj in TRAIN:
        rig = load_stereo_ini(RAW / f"{traj:06d}/StereoCalibrationDVRK.ini")
        kin = load_kinematics(traj)
        n = len(kin["PSM1"]["t"])
        gl, gr = load_gt(traj, "left", n), load_gt(traj, "right", n)
        fr = np.arange(0, n, 2)
        one = np.ones((len(fr), 10))
        G, _, rp = triangulate_seq(rig, gl[fr], gr[fr], one, one)
        G[rp > 3] = np.nan
        for arm, sl in ARMS.items():
            Rk, tk = kin[arm]["R"][fr], kin[arm]["t"][fr]
            piv = G[:, sl][:, 2]
            ok = np.isfinite(piv).all(1)
            R, t = register_rigid_robust(tk[ok], piv[ok])
            pivot_res[arm].append(float(np.median(np.linalg.norm(tk[ok] @ R.T + t - piv[ok], axis=1))))
            Xb = (G[:, sl] - t) @ R                                    # camera -> kinematic base
            loc[arm].append(np.einsum("fji,fkj->fki", Rk, Xb - tk[:, None]))  # base -> tool frame
            q5[arm].append(kin[arm]["q"][fr, 5])

    out = {"fitted_on": list(TRAIN), "arms": {}}
    for arm in ARMS:
        L, q = np.concatenate(loc[arm]), np.concatenate(q5[arm])
        w = L[:, 1]
        okw = np.isfinite(w).all(1)

        def f(p, w=w[okw], q=q[okw]):
            b, Lw, ax = p
            return np.c_[w[:, 0] - ax, w[:, 1] - Lw * np.sin(q + b), w[:, 2] + Lw * np.cos(q + b)].ravel()

        b, Lw, ax = least_squares(f, [0.0, 6.0, 0.0], loss="soft_l1").x
        wres = np.linalg.norm(f([b, Lw, ax]).reshape(-1, 3), axis=1)
        ta, tb = L[:, 3], L[:, 4]
        ok = np.isfinite(ta).all(1) & np.isfinite(tb).all(1)
        mid, half = (ta[ok] + tb[ok]) / 2, (ta[ok] - tb[ok]) / 2
        u = np.median(half / np.linalg.norm(half, axis=1, keepdims=True), 0)
        u /= np.linalg.norm(u)
        c = np.median(mid, 0)
        h = half @ u
        tres = np.linalg.norm(np.concatenate([ta[ok] - (c + h[:, None] * u), tb[ok] - (c - h[:, None] * u)]), axis=1)
        out["arms"][arm] = {
            "wrist": {"b": float(b), "L": float(Lw), "ax": float(ax)},
            "tip_mid": c.tolist(), "open_dir": u.tolist(),
            "h_median": float(np.median(h)), "h_p5_p95": np.percentile(h, [5, 95]).tolist(),
            "diag": {"pivot_oracle_res_mm_per_traj": [round(x, 2) for x in pivot_res[arm]],
                     "wrist_res_mm_median": float(np.median(wres)),
                     "tip_res_mm_median": float(np.median(tres)), "n_frames": int(ok.sum())},
        }
    (ROOT / "configs/tool_geometry.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
