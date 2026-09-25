"""Fit a single softmax temperature on the dev split (pixel NLL). Argmax, and so every Dice/IoU number,
is unchanged; only confidences change. Writes <run>/temperature.json, which eval_seg.py applies."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from surgscene.data import IGNORE, SegDataset, sisvse_frames
from surgscene.models import build_seg_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--pixels-per-frame", type=int, default=4000)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()

    g = torch.Generator().manual_seed(0)
    logits, labels = [], []
    with torch.no_grad():
        for x, y, _ in torch.utils.data.DataLoader(SegDataset("sisvse", sisvse_frames("dev")), 16, num_workers=8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.cuda()).float()
            out, y = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1]), y.cuda().reshape(-1)
            keep = torch.nonzero(y != IGNORE).squeeze(1)
            keep = keep[torch.randperm(len(keep), generator=g)[: args.pixels_per_frame * x.shape[0]].cuda()]
            logits.append(out[keep])
            labels.append(y[keep])
    logits, labels = torch.cat(logits), torch.cat(labels)

    log_t = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    nll_before = F.cross_entropy(logits, labels).item()
    opt.step(closure)
    t = log_t.exp().item()
    nll_after = F.cross_entropy(logits / t, labels).item()
    out = {"temperature": t, "fit_on": "dev", "n_pixels": len(labels), "nll_before": nll_before, "nll_after": nll_after}
    (Path(args.ckpt).parent / "temperature.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
