"""Per-frame perception front end for the stage-2 pipeline (Phases 3b-6).

Native SurgPose frame (1400x986 BGR) -> half resolution, zero-padded to 704x512 -> keypoint
network -> Laplace observations in *native* pixels. The C++ runtime (Phase 6) reproduces
exactly this preprocessing, and its parity tests use goldens produced here.

Stage-2 split. The Phase 2-3 test trajectories 20-33 have been seen, and dev (18-19) shares the
training backgrounds, so tuning on dev alone always says "the shape prior doesn't help". Stage 2
moves one beef (20) and one chicken-thigh (23) trajectory into the tuning set, so every tuned
parameter has seen the background shift. The rest (21, 22, 24-33) is the stage-2 test set.
"""

import numpy as np
import torch

from surgscene.data import normalize
from surgscene.models import build_seg_model
from surgscene.structured import observe

SCALE = 0.5
PAD_WH = (704, 512)
TUNE = [18, 19, 20, 23]
TEST2 = [21, 22] + list(range(24, 34))


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Native frame -> padded half-resolution uint8 BGR (what the network sees, before normalization)."""
    import cv2
    small = cv2.resize(frame_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PAD_WH[1], PAD_WH[0], 3), np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas


def load_kp_model(ckpt: str) -> torch.nn.Module:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_keypoints"], pretrained=False)
    net.load_state_dict(ck["model"])
    return net.cuda().eval()


@torch.no_grad()
def run_batch(net, imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """imgs: padded half-res BGR uint8. Returns kp/sigma in native px, conf."""
    x = torch.stack([normalize(im) for im in imgs]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hm = net(x).float().sigmoid()
    o = observe(hm)
    return {"kp": o.kp / SCALE, "conf": o.conf, "sigma": o.sigma / SCALE}


def load_vit_model(ckpt: str) -> torch.nn.Module:
    """v3 Phase A DINOv2 keypoint network (surgscene.kp_vit)."""
    from surgscene.kp_vit import DinoKeypointNet
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = DinoKeypointNet(cfg["num_keypoints"], layers=cfg["layers"], variant=cfg["variant"], pretrained=False)
    net.load_state_dict(ck["model"])
    net.K = cfg["num_keypoints"]
    return net.cuda().eval()


@torch.no_grad()
def run_batch_vit(net, imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """As run_batch, plus the learned localization std (native px, unscaled)."""
    from surgscene.kp_vit import read_at
    x = torch.stack([normalize(im) for im in imgs]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = net(x).float()
    hm = y[:, : net.K].sigmoid()
    o = observe(hm)
    s = read_at(y[:, net.K:], hm.flatten(2).argmax(-1)).clamp(-6, 12).mul(0.5).exp()
    return {"kp": o.kp / SCALE, "conf": o.conf, "sigma": o.sigma / SCALE, "sigma_learned": s.cpu().numpy() / SCALE}
