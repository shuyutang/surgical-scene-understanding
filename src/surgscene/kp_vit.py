"""v3 Phase A: DINOv2 keypoint network with a learned per-keypoint variance.

Backbone: DINOv2 ViT-S/14 (timm `vit_small_patch14_dinov2.lvd142m`), features from three depths.
Patch-14 tokens are too coarse for sub-pixel keypoints on their own, so a small conv stem on the
image supplies 1/1, 1/2 and 1/4 resolution skips to a U-Net-style decoder. Output at the input
resolution (the v2 frame cache, 704 x 512):
  channels 0..K-1   heatmap logits (sigmoid -> the same heatmaps v2 decodes)
  channels K..2K-1  log-variance map; the value at a keypoint's argmax pixel is log(s^2), s = the
                    predicted localization std in cache px (trained by Gaussian NLL on the actual
                    decoding error, so it predicts error, unlike the Laplace peak width)
Plain conv / attention ops only, for ONNX/TensorRT.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DinoKeypointNet(nn.Module):
    def __init__(self, num_keypoints: int = 10, layers=(5, 8, 11), pretrained: bool = True,
                 img_hw=(512, 704), variant: str = "vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.K, self.layers = num_keypoints, list(layers)
        self.hw = img_hw
        self.pad_hw = tuple(-(-s // PATCH) * PATCH for s in img_hw)
        self.vit = timm.create_model(variant, pretrained=pretrained, img_size=self.pad_hw)
        d = self.vit.embed_dim
        self.stem1 = _block(3, 16)
        self.stem2 = nn.Sequential(_block(16, 32, 2), _block(32, 32))
        self.stem4 = nn.Sequential(_block(32, 64, 2), _block(64, 64))
        self.proj = nn.Sequential(nn.Conv2d(d * len(layers), 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(_block(256 + 64, 128), _block(128, 128))
        self.dec2 = nn.Sequential(_block(128 + 32, 64), _block(64, 64))
        self.dec1 = nn.Sequential(_block(64 + 16, 32), _block(32, 32))
        self.head = nn.Conv2d(32, 2 * num_keypoints, 1)
        nn.init.constant_(self.head.bias[:num_keypoints], -4.6)  # CenterNet prior, as in v2
        nn.init.constant_(self.head.bias[num_keypoints:], 2.0)   # log s^2 = 2: s ~ 2.7 cache px at start

    def backbone_parameters(self):
        return self.vit.parameters()

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("vit.")]

    def forward(self, x):
        H, W = x.shape[-2:]
        xp = F.pad(x, (0, self.pad_hw[1] - W, 0, self.pad_hw[0] - H))
        feats = self.vit.forward_intermediates(xp, indices=self.layers, output_fmt="NCHW", intermediates_only=True)
        f = self.proj(torch.cat(feats, 1))
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        s4 = self.stem4(s2)
        # token grid covers the padded image; crop to the part that lies over the real image
        f = F.interpolate(f, size=(xp.shape[-2] // 4, xp.shape[-1] // 4), mode="bilinear", align_corners=False)
        f = f[..., : s4.shape[-2], : s4.shape[-1]]
        y = self.dec4(torch.cat([f, s4], 1))
        y = self.dec2(torch.cat([F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        y = self.dec1(torch.cat([F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        return self.head(y)


@torch.no_grad()
def decode_torch(hm: torch.Tensor):
    """Argmax + quadratic sub-pixel refinement (as keypoints.decode), on tensors. hm (B, K, H, W)
    probabilities -> kp (B, K, 2) px, conf (B, K), flat argmax index (B, K)."""
    B, K, H, W = hm.shape
    conf, idx = hm.flatten(2).max(-1)
    yi, xi = idx // W, idx % W
    hp = F.pad(hm, (1, 1, 1, 1), mode="replicate")
    bi = torch.arange(B, device=hm.device)[:, None].expand(B, K)
    ki = torch.arange(K, device=hm.device)[None, :].expand(B, K)
    lg = lambda dy, dx: torch.log(hp[bi, ki, yi + 1 + dy, xi + 1 + dx].clamp(min=1e-8))
    c, l, r, u, d = lg(0, 0), lg(0, -1), lg(0, 1), lg(-1, 0), lg(1, 0)
    dx = ((l - r) / (2 * (l - 2 * c + r)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    dy = ((u - d) / (2 * (u - 2 * c + d)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    return torch.stack([xi.float() + dx, yi.float() + dy], -1), conf, idx


def read_at(maps: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """maps (B, K, H, W), idx (B, K) flat pixel index -> (B, K)."""
    return maps.flatten(2).gather(-1, idx[..., None])[..., 0]


def nll_loss(logvar_map, hm_prob, kp_gt, max_err: float = 64.0):
    """Gaussian NLL of the (detached) decoding error under the predicted variance, 2D isotropic:
    |e|^2 / (2 s^2) + log s^2, averaged over labeled keypoints. The heatmap path gets no gradient
    from this term, so the variance head can't trade localization for a smaller loss."""
    kp, _, idx = decode_torch(hm_prob.detach())
    ok = torch.isfinite(kp_gt).all(-1)
    # zero the unlabeled entries *before* the loss: masking a NaN loss afterwards still sends
    # 0 * NaN = NaN back through the variance head
    e2 = torch.where(ok, ((kp - torch.nan_to_num(kp_gt)) ** 2).sum(-1), torch.zeros_like(kp[..., 0]))
    e2 = e2.clamp(max=max_err**2)
    lv = read_at(logvar_map, idx).clamp(-6.0, 12.0)
    loss = e2 / (2 * lv.exp()) + lv
    return loss[ok].mean() if ok.any() else loss.sum() * 0
