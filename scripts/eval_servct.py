"""v3 Phase B, B1: learned stereo vs SGM on SERV-CT (CT-derived disparity ground truth).

  uv run --group stereo python scripts/eval_servct.py      # once, after the pre-registration is committed

16 rectified pairs (720x576) from 2 ex vivo porcine specimens (8 per experiment), da Vinci
endoscope, CT reference (Reference_CT). Valid pixels: GT disparity > 0 and not flagged in
OcclusionL (non-overlap, outside the CT surface, not visible in the right image).
Methods: SGM (proximity.make_sgbm, numDisparities 160 to cover SERV-CT's range) and RAFT-Stereo
(checkpoint selected on SurgPose tune: runs/v3_stereo_tune/results.json). The B3 vertical-offset
rule is applied to both. Depth Z = f B / d with f and B from each pair's P1/P2.
Unit of analysis = stereo pair (16), paired bootstrap, 2,000 replicates, percentile CIs; the
pairs come from only 2 specimens, so per-experiment numbers are reported as well.
Writes runs/v3_servct/{results.json, report.md}.
"""

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_rectification import shift_rows, sift_dy  # noqa: E402

from surgscene.evaluation import ci  # noqa: E402
from surgscene.learned_stereo import LearnedStereo  # noqa: E402
from surgscene.proximity import disparity, make_sgbm  # noqa: E402

DATA = ROOT / "data/servct/SERV-CT"
OUT = ROOT / "runs/v3_servct"
FLAGS_BGR = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]  # red, yellow, blue, green
BAD_PX = 3.0


def valid_mask(gt_disp: np.ndarray, occ_bgr: np.ndarray) -> np.ndarray:
    flagged = np.zeros(gt_disp.shape, bool)
    for c in FLAGS_BGR:
        flagged |= (occ_bgr == np.array(c, np.uint8)).all(-1)
    return (gt_disp > 0) & ~flagged


def load_pair(exp: str, name: str):
    L = cv2.imread(str(DATA / exp / "Left_rectified" / name))
    R = cv2.imread(str(DATA / exp / "Right_rectified" / name))
    gt = cv2.imread(str(DATA / exp / "Reference_CT/Disparity" / name), cv2.IMREAD_UNCHANGED).astype(np.float64) / 256
    occ = cv2.imread(str(DATA / exp / "Reference_CT/OcclusionL" / name))
    cal = json.loads((DATA / exp / "Rectified_calibration" / name.replace(".png", ".json")).read_text())
    P2 = np.array(cal["P2"]["data"]).reshape(3, 4)
    f, B = P2[0, 0], -P2[0, 3] / P2[0, 0]
    return L, R, gt, valid_mask(gt, occ), f, B


def pair_metrics(d, gt, valid, f, B):
    sel = valid & np.isfinite(d) & (d > 0)
    Zp, Zg = f * B / d[sel], f * B / gt[sel]
    return {"depth_mae": float(np.abs(Zp - Zg).mean()), "disp_epe": float(np.abs(d[sel] - gt[sel]).mean()),
            "bad": float((np.abs(d[sel] - gt[sel]) > BAD_PX).mean()), "n": int(sel.sum())}


def boot_paired(diffs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    reps = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    lo, hi = ci(np.asarray(reps))
    return {"point": float(diffs.mean()), "lo": float(lo), "hi": float(hi), "n": len(diffs)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "runs/v3_stereo_tune/results.json").read_text())["selected"]
    net, sgbm, sift = LearnedStereo(sel), make_sgbm(num_disp=160), cv2.SIFT_create(4000)
    rows = []
    for exp in ("Experiment_1", "Experiment_2"):
        for path in sorted(glob.glob(str(DATA / exp / "Left_rectified/*.png"))):
            name = Path(path).name
            L, R, gt, valid, f, B = load_pair(exp, name)
            dy = float(np.median(sift_dy(sift, L, R)))
            Rc = shift_rows(R, dy) if abs(dy) > 0.5 else R  # B3 rule, applied to both methods
            d_sgm, d_net = disparity(sgbm, L, Rc), net.disparity(L, Rc)
            common = valid & np.isfinite(d_sgm) & np.isfinite(d_net)
            rows.append({"exp": exp, "pair": name, "dy": dy, "coverage_sgm": float((valid & np.isfinite(d_sgm)).sum() / valid.sum()),
                         "sgm": pair_metrics(d_sgm, gt, common, f, B), "net": pair_metrics(d_net, gt, common, f, B),
                         "net_all": pair_metrics(d_net, gt, valid, f, B)})
    diff = [r["net"]["depth_mae"] - r["sgm"]["depth_mae"] for r in rows]
    R = {"selected_checkpoint": sel, "pairs": rows,
         "B1_depth_mae_net_minus_sgm": boot_paired(diff),
         "depth_mae_sgm": boot_paired([r["sgm"]["depth_mae"] for r in rows]),
         "depth_mae_net": boot_paired([r["net"]["depth_mae"] for r in rows]),
         "depth_mae_net_all_valid": boot_paired([r["net_all"]["depth_mae"] for r in rows]),
         "bad_net_minus_sgm": boot_paired([r["net"]["bad"] - r["sgm"]["bad"] for r in rows]),
         "epe_net_minus_sgm": boot_paired([r["net"]["disp_epe"] - r["sgm"]["disp_epe"] for r in rows]),
         "coverage_sgm": boot_paired([r["coverage_sgm"] for r in rows])}
    for exp in ("Experiment_1", "Experiment_2"):
        rr = [r for r in rows if r["exp"] == exp]
        R[f"per_experiment_{exp}"] = {"sgm": float(np.mean([r["sgm"]["depth_mae"] for r in rr])),
                                      "net": float(np.mean([r["net"]["depth_mae"] for r in rr]))}
    b1 = R["B1_depth_mae_net_minus_sgm"]
    R["verdict_B1"] = bool(b1["hi"] < 0)
    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    f = lambda d, n=2: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = ["# B1: learned stereo vs SGM on SERV-CT (pre-specified)", "",
         f"RAFT-Stereo checkpoint `{sel}` (selected on SurgPose tune). 16 pairs, paired bootstrap over pairs.", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|",
         f"| B1 | Mean abs depth error, learned − SGM (pixels valid for both), mm | {f(b1)} | UB < 0 | "
         f"{'PASS' if R['verdict_B1'] else 'FAIL'} |", "", "Reported:", "",
         f"- Depth MAE, SGM / learned (common pixels): {f(R['depth_mae_sgm'])} / {f(R['depth_mae_net'])} mm",
         f"- Depth MAE, learned on all valid pixels: {f(R['depth_mae_net_all_valid'])} mm; SGM coverage of valid pixels "
         f"{f(R['coverage_sgm'], 3)}",
         f"- Bad pixels (> {BAD_PX:g} px), learned − SGM: {f(R['bad_net_minus_sgm'], 3)}; disparity EPE, learned − SGM: "
         f"{f(R['epe_net_minus_sgm'])} px",
         f"- Per experiment (depth MAE SGM → learned): Exp 1 {R['per_experiment_Experiment_1']['sgm']:.2f} → "
         f"{R['per_experiment_Experiment_1']['net']:.2f} mm; Exp 2 {R['per_experiment_Experiment_2']['sgm']:.2f} → "
         f"{R['per_experiment_Experiment_2']['net']:.2f} mm",
         f"- Vertical offset measured per pair (B3): {', '.join(f'{r['dy']:+.2f}' for r in rows)} px"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
