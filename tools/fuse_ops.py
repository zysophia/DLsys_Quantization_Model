import needle as ndl
import needle.nn as nn


def _run_fuse_conv_bn_relu(module):
    """Fuse Convolution BatchNorm and ReLU helper"""
    conv = module.conv
    bn = module.bn

    fused = nn.FuseConv(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.bias,
        conv.device,
        conv.dtype
    )

    w_conv = conv.weight.detach().\
        transpose((0,3)).transpose((1,2)).transpose((0,1)).\
            reshape((conv.out_channels, -1))
    w_bn = ndl.diag(bn.weight / ndl.sqrt(bn.eps + bn.running_var))

    fused_weight = ndl.matmul(w_bn, w_conv)
    tmp_shape = [
        fused.weight.shape[3],
        fused.weight.shape[2],
        fused.weight.shape[0],
        fused.weight.shape[1],
    ]
    fused_weight = fused_weight.reshape(tmp_shape)
    fused.weight.cached_data = fused_weight.realize_cached_data().permute((2,3,1,0))

    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = ops.zeros(conv.weight.shape[3], device=ndl.cpu())
    b_conv = ndl.matmul(w_bn, b_conv.reshape((-1, 1))).reshape(-1)
    b_bn = bn.bias - bn.weight * bn.running_mean / ndl.sqrt(bn.running_var + bn.eps)
    fused_b = b_conv + b_bn
    fused.bias.cached_data = fused_b.realize_cached_data()

    return fused


def _fuse_conv_bn_relu(model):
    """Fuse Convolution BatchNorm and ReLU helper"""
    if not isinstance(model, nn.Module):
        return model

    if isinstance(model, nn.ConvBatchNormReLU):
        model = _run_fuse_conv_bn_relu(model)
        return model

    keys = list(model.__dict__.keys())
    for name in keys:
        if isinstance(getattr(model, name), nn.ConvBatchNormReLU):
            setattr(model, name, _run_fuse_conv_bn_relu(getattr(model, name)))
        elif isinstance(getattr(model, name), (list, tuple)):
            new_modules = []
            for sub_model in getattr(model, name):
                if isinstance(sub_model, nn.ConvBatchNormReLU):
                    sub_model = _run_fuse_conv_bn_relu(sub_model)
                new_modules.append(sub_model)
            setattr(model, name, tuple(new_modules))
        elif isinstance(getattr(model, name), nn.Module):
            setattr(model, name, _fuse_conv_bn_relu(getattr(model, name)))
    return model


def fuse_conv_bn_relu(model):
    """Fuse Convolution BatchNorm and ReLU api"""
    return _fuse_conv_bn_relu(model)
