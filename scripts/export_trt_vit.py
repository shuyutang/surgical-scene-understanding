"""v3 Phase E: export the DINOv2 keypoint deploy graph to ONNX, build TensorRT FP32/FP16 engines.

  uv run --group deploy python scripts/export_trt_vit.py runs/<kp_vit_run>/best.pt

Preprocessing and decoding are pinned to FP32 inside the FP16 engine (trt_runner.build_engine
fp32_outside): without that, the argmax-index arithmetic overflows in FP16 and every keypoint is NaN.

Writes runs/deploy_vit/kp_vit_stereo.onnx (opset 17, static batch 2, uint8 input) and
kp_vit_stereo_{fp32,fp16}.engine. Outputs: kp (2,10,2), conf (2,10), sigma (2,10) Laplace,
sigma_learned (2,10), native px.
"""

import argparse
import json
from pathlib import Path

import torch

from surgscene.deploy import DeployModelVit
from surgscene.frontend import load_vit_model
from surgscene.runinfo import git_state
from surgscene.trt_runner import FP32_TYPES_VIT, build_engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/deploy_vit"
OPSET = 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    net = load_vit_model(args.ckpt)
    model = DeployModelVit(net).cuda().eval()
    dummy = torch.zeros(2, 986, 1400, 3, dtype=torch.uint8, device="cuda")
    onnx_path = OUT / "kp_vit_stereo.onnx"
    torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=OPSET, dynamo=False,
                      input_names=["frames"], output_names=["kp", "conf", "sigma", "sigma_learned"])
    print("wrote", onnx_path)
    for prec in ["fp32", "fp16"]:
        build_engine(onnx_path, OUT / f"kp_vit_stereo_{prec}.engine", fp16=prec == "fp16", fp32_outside="/net/",
                     fp32_types=FP32_TYPES_VIT)
        print("built", prec)
    import tensorrt as trt
    (OUT / "export_info.json").write_text(json.dumps(
        {"checkpoint": args.ckpt, "opset": OPSET, "tensorrt": trt.__version__, "torch": torch.__version__,
         "input": "frames uint8 (2, 986, 1400, 3) BGR",
         "outputs": ["kp (2,10,2) native px", "conf (2,10)", "sigma (2,10)", "sigma_learned (2,10) native px"],
         "git": git_state()}, indent=1))


if __name__ == "__main__":
    main()
