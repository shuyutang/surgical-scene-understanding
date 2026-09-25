import segmentation_models_pytorch as smp
import torch.nn as nn


def build_seg_model(arch: str, encoder: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Plain conv architectures only: every op here exports cleanly to ONNX/TensorRT."""
    cls = {"unet": smp.Unet, "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
    return cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=num_classes)
