import needle
import needle.nn as nn
from .quantize_weights import quantize_conv
from .fuse_ops import fuse_conv_bn_relu


def _quant_model(model):
    if not isinstance(model, nn.Module):
        return model

    if isinstance(model, (nn.Conv, nn.FuseConv)):
        model = quantize_conv(model)
        return model

    keys = list(model.__dict__.keys())
    for name in keys:
        if isinstance(getattr(model, name), (nn.Conv, nn.FuseConv)):
            setattr(obj, name, value)


def quant_model(model, fuse=False):
    if fuse:
        model = fuse_conv_bn_relu(model)
    
    model = _quant_model(model)

    return model