"""v3 Phase E, E1 (R6): TensorRT FP16 DINOv2 engine vs the PyTorch FP32 deploy graph, accuracy
against GT on test2 (left eye, every 5th frame), paired and trajectory-clustered, as v2's D2.

  uv run --group deploy python scripts/eval_e1_vit.py      # once, after the pre-registration

Writes runs/deploy_vit/e1_test2.json and prints the verdict.
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from surgscene.deploy import DeployModelVit  # noqa: E402
from surgscene.frontend import TEST2, load_vit_model  # noqa: E402
from surgscene.stage2 import load_obs  # noqa: E402
from surgscene.trt_runner import TrtRunner  # noqa: E402
from surgscene.evaluation import boot  # noqa: E402
from surgscene.video import frames  # noqa: E402

DEP = ROOT / "runs/deploy_vit"
MARGIN = 0.25


def main():
    info = json.loads((DEP / "export_info.json").read_text())
    ref = DeployModelVit(load_vit_model(info["checkpoint"])).cuda().eval()
    t16 = TrtRunner(DEP / "kp_vit_stereo_fp16.engine")
    diff, e16, e32 = [], [], []
    for traj in TEST2:
        gl = load_obs(traj, "left")["gt"]
        d, a, b = [], [], []
        for t, pair in frames(traj, 5):
            with torch.no_grad():
                kp32 = ref(torch.from_numpy(pair).cuda())[0].cpu().numpy()[0]
            kp16 = t16(pair)["kp"][0]
            x32 = np.linalg.norm(kp32 - gl[t], axis=-1)
            x16 = np.linalg.norm(kp16 - gl[t], axis=-1)
            d.append(x16 - x32)
            a.append(x16)
            b.append(x32)
        diff.append(np.concatenate(d))
        e16.append(np.concatenate(a))
        e32.append(np.concatenate(b))
        print(traj, f"{np.nanmean(diff[-1]):+.3f}", flush=True)
    R = {"E1_fp16_minus_fp32_px": boot(diff), "fp16_mean_err": boot(e16), "fp32_mean_err": boot(e32), "margin": MARGIN}
    R["verdict_E1"] = bool(R["E1_fp16_minus_fp32_px"]["hi"] <= MARGIN)
    (DEP / "e1_test2.json").write_text(json.dumps(R, indent=1))
    d = R["E1_fp16_minus_fp32_px"]
    print(f"E1: FP16 − FP32 mean error {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] px; criterion UB ≤ +{MARGIN} -> "
          f"{'PASS' if R['verdict_E1'] else 'FAIL'}; FP32 {R['fp32_mean_err']['point']:.2f}, FP16 {R['fp16_mean_err']['point']:.2f} px")


if __name__ == "__main__":
    main()
