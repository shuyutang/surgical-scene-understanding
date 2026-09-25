"""Train the SurgPose keypoint heatmap model. Model selection on dev (trajectories 18-19) only."""

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
from surgscene.keypoints import KeypointDataset, decode, focal_loss, load_records
from surgscene.models import build_seg_model
from surgscene.runinfo import write_runinfo

ROOT = Path(__file__).resolve().parents[1]
NATIVE = 2.0  # cache px -> native px


@torch.no_grad()
def dev_pck(model, loader, thr_native=10.0) -> tuple[float, float]:
    model.eval()
    errs = []
    for x, _, kp, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm = model(x.cuda()).float().sigmoid()
        pred, _ = decode(hm)
        errs.append(np.linalg.norm(pred - kp.numpy(), axis=-1) * NATIVE)
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

    train_dl = DataLoader(KeypointDataset(load_records("train"), augment=True), cfg["batch_size"], shuffle=True,
                          num_workers=cfg["workers"], drop_last=True, pin_memory=True, persistent_workers=True,
                          worker_init_fn=worker_init)
    dev_dl = DataLoader(KeypointDataset(load_records("dev")), 16, num_workers=cfg["workers"])

    model = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_keypoints"]).cuda()
    # start heatmap logits low (CenterNet prior) so early training isn't swamped by negatives
    torch.nn.init.constant_(model.segmentation_head[0].bias, -4.6)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total, warmup = cfg["epochs"] * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best, log = -1.0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0, losses = time.time(), []
        for x, hm, _, _ in train_dl:
            x, hm = x.cuda(non_blocking=True), hm.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = focal_loss(logits.float(), hm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            pck, med = dev_pck(model, dev_dl)
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
