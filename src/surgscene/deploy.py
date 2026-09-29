"""Phase 6: the deployable graph = preprocessing + keypoint network + heatmap decoding, end to end.

    uint8 BGR frames (B, 986, 1400, 3)  ->  kp (B, 10, 2) native px, conf (B, 10), sigma (B, 10) native px

Folding preprocessing and decoding into the exported graph means TensorRT runs all of it on the
GPU (no custom CUDA kernels, no nvcc), the host uploads 4 MB of uint8 per frame, and it downloads
30 numbers per image instead of a 14 MB heatmap. Every op here is plain ONNX (AveragePool, Pad,
ReduceMax/ArgMax, GatherElements, elementwise), which TensorRT supports natively.

Differences from the Python front end (frontend.preprocess + structured.observe), all measured by
the parity tests: resizing is done in float (cv2's INTER_AREA rounds to uint8 first), and the
network runs in FP32/FP16 instead of bf16 autocast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD_WH = (704, 512)
SCALE = 0.5


class DeployModel(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        # BGR input -> RGB normalization, as per-channel constants in BGR order
        self.register_buffer("mean", torch.tensor(MEAN[::-1]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD[::-1]).view(1, 3, 1, 1) * 255)

    def forward(self, frames: torch.Tensor):
        x = frames.float().permute(0, 3, 1, 2)                   # (B, 3, 986, 1400) BGR; cast first (TRT: no uint8 intermediates)
        x = F.avg_pool2d(x, 2)                                    # exact INTER_AREA at 0.5 (float, no rounding)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, PAD_WH[0] - w, 0, PAD_WH[1] - h))        # black padding, as in the frame cache
        x = ((x - self.mean) / self.std).flip(1)                  # normalize, BGR -> RGB
        hm = self.net(x).sigmoid()
        return decode_graph(hm)


def decode_graph(hm: torch.Tensor):
    """Graph-friendly version of keypoints.decode + structured.observe (sigma). hm: (B, K, H, W)."""
    B, K, H, W = hm.shape
    flat = hm.flatten(2)
    conf, idx = flat.max(-1)
    fi = idx.float()
    ys = torch.floor(fi / W)
    xs = fi - ys * W

    # sub-pixel: quadratic fit of the log heatmap on the replicate-padded map (keypoints.decode)
    hp = F.pad(hm, (1, 1, 1, 1), mode="replicate").flatten(2)
    Wp = W + 2

    def g(src, yy, xx, stride, lo):
        i = (yy * stride + xx).long()
        return torch.log(torch.gather(src, 2, i.unsqueeze(-1)).squeeze(-1).clamp(min=lo))

    yi, xi = ys + 1, xs + 1
    c, l, r = g(hp, yi, xi, Wp, 1e-8), g(hp, yi, xi - 1, Wp, 1e-8), g(hp, yi, xi + 1, Wp, 1e-8)
    u, d = g(hp, yi - 1, xi, Wp, 1e-8), g(hp, yi + 1, xi, Wp, 1e-8)
    dx = ((l - r) / (2 * (l - 2 * c + r)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    dy = ((u - d) / (2 * (u - 2 * c + d)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    kp = torch.stack([xs + dx, ys + dy], -1) / SCALE

    # Laplace sigma from the log-heatmap curvature at the (interior-clamped) integer peak (structured.observe)
    yc, xc = ys.clamp(1, H - 2), xs.clamp(1, W - 2)
    c0 = g(flat, yc, xc, W, 1e-6)
    dxx = g(flat, yc, xc - 1, W, 1e-6) - 2 * c0 + g(flat, yc, xc + 1, W, 1e-6)
    dyy = g(flat, yc - 1, xc, W, 1e-6) - 2 * c0 + g(flat, yc + 1, xc, W, 1e-6)
    curv = -(dxx + dyy) / 2
    sigma = (1 / curv.clamp(min=1e-3)).sqrt().clamp(1.0, 20.0) / SCALE
    return kp, conf, sigma


class DeployModelVit(DeployModel):
    """v3 Phase A: the same deploy graph around the DINOv2 keypoint network. The network emits
    heatmap logits and a log-variance map; the learned std is read at each keypoint's argmax."""

    def forward(self, frames: torch.Tensor):
        x = frames.float().permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, 2)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, PAD_WH[0] - w, 0, PAD_WH[1] - h))
        x = ((x - self.mean) / self.std).flip(1)
        y = self.net(x)
        K = y.shape[1] // 2
        hm = y[:, :K].sigmoid()
        kp, conf, sigma = decode_graph(hm)
        idx = hm.flatten(2).argmax(-1, keepdim=True)
        lv = torch.gather(y[:, K:].flatten(2), 2, idx).squeeze(-1).clamp(-6.0, 12.0)
        return kp, conf, sigma, (0.5 * lv).exp() / SCALE
