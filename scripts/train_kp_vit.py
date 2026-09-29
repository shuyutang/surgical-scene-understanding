"""v3 Phase A: train the DINOv2 keypoint network. Model selection on dev (trajectories 18-19) only.

  uv run python scripts/train_kp_vit.py configs/kp_vit_s.yaml

Same data, augmentation (de-centered occluders + distractors), schedule shape and dev selection as
the v2 U-Net (scripts/train_kp.py). Loss = focal (argmax positives, v3 A0) + nll_weight * Gaussian
NLL of the decoding error under the predicted variance.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from surgscene.data import worker_init
from surgscene.keypoints import KeypointDataset, focal_loss, load_records
from surgscene.kp_vit import DinoKeypointNet, decode_torch, nll_loss
from surgscene.runinfo import write_runinfo

ROOT = Path(__file__).resolve().parents[1]

NATIVE = 2.0


@torch.no_grad()
def dev_pck(model, loader, K, thr_native=10.0):
    model.eval()
    errs = []
    for x, _, kp, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.cuda()).float()
        pred, _, _ = decode_torch(out[:, :K].sigmoid())
        errs.append((torch.linalg.norm(pred.cpu() - kp, dim=-1) * NATIVE).numpy())
    e = np.concatenate(errs)
    e = e[np.isfinite(e)]
    model.train()
    return float((e <= thr_native).mean()), float(np.median(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.manual_seed(cfg["seed"])
    run_dir = ROOT / "runs" / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    write_runinfo(run_dir, cfg)
    K = cfg["num_keypoints"]

    train_dl = DataLoader(KeypointDataset(load_records("train"), augment=True, decentered_aug=True), cfg["batch_size"],
                          shuffle=True, num_workers=cfg["workers"], drop_last=True, pin_memory=True,
                          persistent_workers=True, worker_init_fn=worker_init)
    dev_dl = DataLoader(KeypointDataset(load_records("dev")), 16, num_workers=cfg["workers"])

    model = DinoKeypointNet(K, layers=cfg["layers"], variant=cfg["variant"]).cuda()
    opt = torch.optim.AdamW([{"params": model.backbone_parameters(), "lr": cfg["lr_backbone"]},
                             {"params": model.head_parameters(), "lr": cfg["lr"]}], weight_decay=cfg["weight_decay"])
    total, warmup = cfg["epochs"] * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best, log = -1.0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0, lf, ln = time.time(), [], []
        for x, hm, kp, _ in train_dl:
            x, hm, kp = x.cuda(non_blocking=True), hm.cuda(non_blocking=True), kp.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            out = out.float()
            l_focal = focal_loss(out[:, :K], hm, pos_mode="argmax")
            l_nll = nll_loss(out[:, K:], out[:, :K].sigmoid(), kp)
            loss = l_focal + cfg["nll_weight"] * l_nll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            lf.append(l_focal.item())
            ln.append(l_nll.item())
        entry = {"epoch": epoch, "focal": float(np.mean(lf)), "nll": float(np.mean(ln)), "sec": round(time.time() - t0, 1)}
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            pck, med = dev_pck(model, dev_dl, K)
            entry.update(dev_pck10=pck, dev_median_err_px=med)
            if pck > best:
                best = pck
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "dev_pck10": pck},
                           run_dir / "best.pt")
        log.append(entry)
        print(json.dumps(entry), flush=True)
        (run_dir / "log.json").write_text(json.dumps(log, indent=1))
    print(f"best dev PCK@10px {best:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
