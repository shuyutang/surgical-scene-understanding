"""v4: instrument-type geometry library, from train trajectories (0-17) only.

  uv run python scripts/v4_tool_library.py

v3 fitted one geometry per arm over all train trajectories. SurgPose mixes instrument types (tip
offset 9-29 mm), and on long-jaw tools the online fit of the tip offset collapses toward that
standard-tool prior: where the jaws point along the viewing ray, 2D data can't see their length
(tune 18 PSM3: fitted 17.8 mm vs 29.2 mm GT). A real system knows the instrument type (da Vinci
instruments identify themselves); here the type has to be recognised from the images, and the
geometry comes from a per-type library.

Per train arm-trajectory: oracle hand-eye from the GT-triangulated pivot (as fit_tool_geometry.py),
GT keypoints in the tool frame, tip-midpoint offset c = median. Type = "long" if |c| > LONG_MM
(the v3 pre-specified subgroup threshold), else "standard". Per type, pooled over both arms:
tip_mid, open_dir, h per arm (the two arms aren't pooled: their jaw-opening sign conventions
differ); the wrist model stays per arm from configs/tool_geometry.json (it doesn't depend on the jaw).
Train has one long-jaw trajectory (17, both arms), so the long type rests on one example per arm.
Writes configs/v4_tool_library.json.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from surgscene.fusion import ARMS, load_kinematics  # noqa: E402
from surgscene.geometry import load_stereo_ini, register_rigid_robust  # noqa: E402
from surgscene.stage2 import triangulate_seq  # noqa: E402
from surgscene.rectification import load_gt  # noqa: E402
from surgscene.stage2 import RAW  # noqa: E402

TRAIN = range(0, 18)
LONG_MM = 15.0


def main():
    per = {}
    mids = {(typ, arm): [] for typ in ("standard", "long") for arm in ARMS}
    halves = {k: [] for k in mids}
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
            loc = np.einsum("fji,fkj->fki", Rk, (G[:, sl] - t) @ R - tk[:, None])
            ta, tb = loc[:, 3], loc[:, 4]
            okt = np.isfinite(ta).all(1) & np.isfinite(tb).all(1)
            mid, half = (ta[okt] + tb[okt]) / 2, (ta[okt] - tb[okt]) / 2
            c = np.median(mid, 0)
            typ = "long" if np.linalg.norm(c) > LONG_MM else "standard"
            per[f"{traj}/{arm}"] = {"tip_offset_mm": round(float(np.linalg.norm(c)), 2), "type": typ,
                                    "scatter_mm": round(float(np.median(np.linalg.norm(mid - c, axis=1))), 2)}
            mids[(typ, arm)].append(mid)
            halves[(typ, arm)].append(half)
            print(traj, arm, per[f"{traj}/{arm}"], flush=True)
    base = json.loads((ROOT / "configs/tool_geometry.json").read_text())
    out = {"fitted_on": list(TRAIN), "long_mm": LONG_MM, "arm_trajectories": per, "types": {}}
    for (typ, arm) in mids:
        if not mids[(typ, arm)]:
            continue
        M, H = np.concatenate(mids[(typ, arm)]), np.concatenate(halves[(typ, arm)])
        u = np.median(H / np.linalg.norm(H, axis=1, keepdims=True), 0)
        u /= np.linalg.norm(u)
        h = H @ u
        c = np.median(M, 0)
        out["types"].setdefault(typ, {})[arm] = {"n_trajectories": len(mids[(typ, arm)]), "tip_mid": c.tolist(), "open_dir": u.tolist(),
                             "h_median": float(np.median(h)), "tip_offset_mm": float(np.linalg.norm(c)),
                             "scatter_mm_median": float(np.median(np.linalg.norm(M - c, axis=1)))}
    # jaw-length spread across instruments of a type, pooled over arms (long: 2 values, from train 17)
    for typ in out["types"]:
        Ls = [v["tip_offset_mm"] for v in per.values() if v["type"] == typ]
        out["types"][typ]["length_sd_mm"] = float(np.std(Ls, ddof=1))
    out["wrist"] = {arm: base["arms"][arm]["wrist"] for arm in ARMS}
    (ROOT / "configs/v4_tool_library.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["types"], indent=1))


if __name__ == "__main__":
    main()
