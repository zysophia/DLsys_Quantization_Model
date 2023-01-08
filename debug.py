#!/usr/bin/env python
# author: fan.mo
# email: fmo@voxelcloud.net.cn
# -*- coding: UTF-8 -*-

import sys
sys.path.append('python')

import torch
import torchvision

import needle as ndl
import needle.nn as nn
from apps.models import ResNet9
from tools.fuse_ops import fuse_conv_bn_relu

net = ResNet9(ndl.cpu(), 'float32')
a = ndl.init.rand(1, 3, 32, 32, device=ndl.cpu())
b = net(a)
print(b.shape)

print(net.named_parameters().keys())
net = fuse_conv_bn_relu(net)
print(net.named_parameters().keys())


def fuse(conv, bn):

    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    _conv = nn.Conv(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        bias=False,
        device=ndl.cpu()
    )
    _bn = nn.BatchNorm2d(
        bn.weight.shape[0],
        eps=bn.eps,
        device=ndl.cpu()
    )
    _conv.weight.cached_data = ndl.NDArray(conv.weight.numpy(), device=ndl.cpu())
    _bn.weight.cached_data = ndl.NDArray(bn.weight.numpy(), device=ndl.cpu())
    _bn.running_mean.cached_data = ndl.NDArray(bn.running_mean.numpy(), device=ndl.cpu())
    _bn.running_var.cached_data = ndl.NDArray(bn.running_var.numpy(), device=ndl.cpu())
    _fused = nn.Conv(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        bias=True,
        device=ndl.cpu()
    )

    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    
    _w_conv = _conv.weight.detach().reshape((_conv.out_channels, -1))
    _w_bn = ndl.diag(_bn.weight / ndl.sqrt(_bn.eps + _bn.running_var))

    fused.weight.copy_( torch.mm(w_bn, w_conv).view(fused.weight.size()) )
    _fused.weight.cached_data = ndl.matmul(_w_bn, _w_conv).reshape(_fused.weight.shape)

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
                            torch.sqrt(bn.running_var + bn.eps)
                        )
    fused.bias.copy_( b_conv + b_bn )

    return fused

# Testing
# we need to turn off gradient calculation because we didn't write it
torch.set_grad_enabled(False)
x = torch.randn(16, 3, 256, 256)
resnet18 = torchvision.models.resnet18(pretrained=True)
# removing all learning variables, etc
resnet18.eval()
model = torch.nn.Sequential(
    resnet18.conv1,
    resnet18.bn1
)
f1 = model.forward(x)
fused = fuse(model[0], model[1])
f2 = fused.forward(x)
d = (f1 - f2).mean().item()
print("error:",d)