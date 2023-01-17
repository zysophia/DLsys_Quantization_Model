import needle
import needle.nn as nn
from tools.quantize_weights import quantize_conv
from .fuse_ops import fuse_conv_bn_relu


def _quant_model(model):
    if not isinstance(model, nn.Module):
        return model

    if isinstance(model, (nn.Conv, nn.FuseConv)):
        quant_weight, scales = quantize_conv(getattr(model, name))
        stride = model.stride
        device = model.device
        bias = model.bias
        model = nn.QConv(quant_weight, scales, stride=stride, bias=bias, device=device)

        return model

    keys = list(model.__dict__.keys())
    for name in keys:
        if isinstance(getattr(model, name), (nn.Conv, nn.FuseConv)):
            quant_weight, scales = quantize_conv(getattr(model, name))
            stride = getattr(model, name).stride
            device = getattr(model, name).device
            bias = getattr(model, name).bias
            setattr(
                model,
                name,
                nn.QConv(quant_weight, scales, stride=stride, bias=bias, device=device),
            )
        elif isinstance(getattr(model, name), (list, tuple)):
            new_modules = []
            for sub_model in getattr(model, name):
                if isinstance(sub_model, (nn.Conv, nn.FuseConv)):
                    quant_weight, scales = quantize_conv(sub_model)
                    stride = sub_model.stride
                    device = sub_model.device
                    bias = sub_model.bias
                    sub_model = nn.QConv(quant_weight, scales)
                new_modules.append(sub_model)
            setattr(model, name, tuple(new_modules))
        elif isinstance(getattr(model, name), nn.Module):
            setattr(model, name, _quant_model(getattr(model, name)))

    return model


def quant_model(model, fuse=False):
    if fuse:
        model = fuse_conv_bn_relu(model)

    model = _quant_model(model)

    return model
