"""Train the SISVSE segmentation model. Model selection uses the dev split only; test stays untouched."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from surgscene.data import IGNORE, SegDataset, sisvse_frames, worker_init
from surgscene.models import build_seg_model
from surgscene.runinfo import write_runinfo

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def dev_miou(model, loader, num_classes, device) -> tuple[float, np.ndarray]:
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        valid = y != IGNORE
        conf += torch.bincount(y[valid] * num_classes + pred[valid], minlength=num_classes**2).view(num_classes, num_classes)
    conf = conf.double()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag())
    present = conf.sum(1) > 0
    model.train()
    return iou[present].mean().item(), iou.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    device = "cuda"
    run_dir = ROOT / "runs" / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    write_runinfo(run_dir, cfg)

    train_ds = SegDataset("sisvse", sisvse_frames("train"), augment=True)
    dev_ds = SegDataset("sisvse", sisvse_frames("dev"))
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
                          pin_memory=True, persistent_workers=True, worker_init_fn=worker_init)
    dev_dl = DataLoader(dev_ds, 16, num_workers=cfg["workers"], pin_memory=True)

    model = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_classes"]).to(device).to(memory_format=torch.channels_last)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(train_dl)
    warmup = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best, log = -1.0, []
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0, losses = time.time(), []
        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits, y, ignore_index=IGNORE) + cfg["dice_weight"] * dice(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            miou, _ = dev_miou(model, dev_dl, cfg["num_classes"], device)
            entry["dev_miou"] = miou
            if miou > best:
                best = miou
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "dev_miou": miou},
                           run_dir / "best.pt")
        log.append(entry)
        print(json.dumps(entry), flush=True)
        (run_dir / "log.json").write_text(json.dumps(log, indent=1))
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": cfg["epochs"]}, run_dir / "last.pt")
    print(f"best dev mIoU {best:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
