import needle as ndl
import needle.nn as nn


def _run_fuse_conv_bn_relu(module):
    """Fuse Convolution BatchNorm and ReLU helper"""
    conv = module.conv
    bn = module.bn

    fused = nn.Conv(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.bias,
        conv.device,
        conv.dtype
    )

    w_conv = conv.weight.detach().reshape((conv.out_channels, -1))
    w_bn = ndl.diag(bn.weight / ndl.sqrt(bn.eps + bn.running_var))

    fused.weight.cached_data = ndl.matmul(w_bn, w_conv).reshape(fused.weight.shape)

    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = ops.zeros(conv.weight.shape[0])
    b_conv = ndl.matmul(w_bn, b_conv.reshape((-1, 1))).reshape((-1))
    b_bn = bn.bias - bn.weight * bn.running_mean / ndl.sqrt(bn.running_var + bn.eps)
    fused.bias.realized_cache_data = b_conv + b_bn

    return fused

def _fuse_conv_bn_relu(model):
    """Fuse Convolution BatchNorm and ReLU helper"""
    if not isinstance(model, nn.Module):
        return

    keys = list(model.__dict__.keys())
    for name in keys:
        if isinstance(getattr(model, name), nn.ConvBatchNormReLU):
            setattr(model, name, _run_fuse_conv_bn_relu(getattr(model, name)))
            # delattr(model, name)
        elif isinstance(getattr(model, name), (list, tuple)):
            new_modules = []
            for sub_model in getattr(model, name):
                if isinstance(sub_model, nn.ConvBatchNormReLU):
                    sub_model = _run_fuse_conv_bn_relu(sub_model)
                new_modules.append(sub_model)
            setattr(model, name, tuple(new_modules))
        elif isinstance(getattr(model, name), nn.Module):
            _fuse_conv_bn_relu(getattr(model, name))


def fuse_conv_bn_relu(model):
    """Fuse Convolution BatchNorm and ReLU api"""
    _fuse_conv_bn_relu(model)

    return model