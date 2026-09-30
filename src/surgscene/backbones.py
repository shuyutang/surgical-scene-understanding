"""Frozen image backbones for the scene-semantics track (GraSP), shared by the frame-level and the
instance-level (crop) feature scripts.

  resnet50     torchvision ResNet-50, ImageNet (IMAGENET1K_V2), global-average-pooled 2048-d
  dinov2_b14   timm vit_base_patch14_dinov2 (LVD-142M), [CLS, mean patch token] 1536-d
  endossl_l16  EndoSSL ViT-L/16 (MSN on private laparoscopy video; Hirsch et al., MICCAI 2023), the
               PyTorch conversion released with SurgVISTA (third_party/endossl/surgvista_teacher.pth;
               its weights equal the official JAX checkpoint). It takes raw 0-255 RGB, no
               normalisation: that reproduces the official TF SavedModel (cosine 0.9998 on the CLS
               output). [CLS, mean patch token] after the final norm, 2048-d
All take a 224 x 224 input.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BACKBONES = ["resnet50", "dinov2_b14", "endossl_l16"]


def preprocess(rgb: np.ndarray, backbone: str) -> torch.Tensor:
    """uint8 RGB (H, W, 3), any size -> (3, 224, 224) float tensor for the backbone."""
    im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return torch.from_numpy((im if backbone == "endossl_l16" else (im / 255.0 - MEAN) / STD).transpose(2, 0, 1))


def endossl():
    """EndoSSL ViT-L/16 in timm: separate q/k/v -> qkv, 14x14 position embedding without a CLS entry."""
    import timm
    sd = torch.load(ROOT / "third_party/endossl/surgvista_teacher.pth", map_location="cpu")
    net = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, no_embed_class=True, global_pool="")
    new = {}
    for k, v in sd.items():
        if ".attn.q." in k:
            new[k.replace(".attn.q.", ".attn.qkv.")] = torch.cat([v, sd[k.replace(".q.", ".k.")], sd[k.replace(".q.", ".v.")]])
        elif ".attn.k." not in k and ".attn.v." not in k:
            new[k] = v
    net.load_state_dict(new, strict=True)
    return net


def build(backbone: str):
    """-> (module, forward function: (B, 3, 224, 224) -> (B, D))."""
    if backbone == "resnet50":
        import torchvision
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        return net, lambda x: net(x)
    if backbone == "endossl_l16":
        net = endossl()
    else:
        import timm
        net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0)

    def fwd(x):
        t = net.forward_features(x)  # (B, 1 + N, D), final norm applied
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
    return net, fwd
