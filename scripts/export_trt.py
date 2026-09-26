"""Phase 6: export the deploy graph (preprocess + network + decode) to ONNX, build TensorRT engines.

  uv run --group deploy python scripts/export_trt.py runs/<kp_run>/best.pt

Writes runs/deploy/kp_stereo.onnx (opset 17, static batch 2 = one stereo pair, uint8 input)
and runs/deploy/kp_stereo_{fp32,fp16}.engine.
"""

import argparse
import json
from pathlib import Path

import torch

from surgscene.deploy import DeployModel
from surgscene.frontend import load_kp_model
from surgscene.runinfo import git_state
from surgscene.trt_runner import build_engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/deploy"
OPSET = 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = DeployModel(load_kp_model(args.ckpt)).cuda().eval()
    dummy = torch.zeros(2, 986, 1400, 3, dtype=torch.uint8, device="cuda")
    onnx_path = OUT / "kp_stereo.onnx"
    torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=OPSET, dynamo=False,
                      input_names=["frames"], output_names=["kp", "conf", "sigma"])
    print("wrote", onnx_path)
    for prec in ["fp32", "fp16"]:
        build_engine(onnx_path, OUT / f"kp_stereo_{prec}.engine", fp16=prec == "fp16")
        print("built", prec)
    import tensorrt as trt
    (OUT / "export_info.json").write_text(json.dumps(
        {"checkpoint": args.ckpt, "opset": OPSET, "tensorrt": trt.__version__, "torch": torch.__version__,
         "input": "frames uint8 (2, 986, 1400, 3) BGR", "outputs": ["kp (2,10,2) native px", "conf (2,10)", "sigma (2,10)"],
         "git": git_state()}, indent=1))


if __name__ == "__main__":
    main()
