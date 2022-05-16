import torch
import torch.nn as nn
from PIL import Image


def normalized_tanh(x, inplace: bool = False):
    return 0.5 * x.tanh() + 0.5


class NormalizedTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(NormalizedTanh, self).__init__()

    def forward(self, x):
        return normalized_tanh(x)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Taken from torchvision
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


def split_normalization_params(model: nn.Module, norm_classes=None):
    # Adapted from https://github.com/facebookresearch/ClassyVision/blob/659d7f78/classy_vision/generic/util.py#L501
    if not norm_classes:
        norm_classes = [nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm]

    for t in norm_classes:
        if not issubclass(t, nn.Module):
            raise ValueError(f"Class {t} is not a subclass of nn.Module.")

    classes = tuple(norm_classes)

    norm_params = []
    other_params = []
    for module in model.modules():
        if next(module.children(), None):
            other_params.extend(
                p for p in module.parameters(recurse=False) if p.requires_grad
            )
        elif isinstance(module, classes):
            norm_params.extend(p for p in module.parameters() if p.requires_grad)
        else:
            other_params.extend(p for p in module.parameters() if p.requires_grad)
    return norm_params, other_params
