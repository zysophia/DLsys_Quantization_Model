import sys

sys.path.append("python")
sys.path.append("tools")
import needle as ndl
import needle.nn as nn
import numpy as np
from fuse_ops import fuse_conv_bn_relu

device = ndl.cpu()
dtype = "float32"


def test_single():
    model = nn.ConvBatchNormReLU(3, 16, 3, 3, device=device, dtype=dtype)
    model.eval()
    inputs = ndl.Tensor(np.random.randn(1, 3, 32, 32), device=device)
    ori_out = model(inputs)

    fuse_model = fuse_conv_bn_relu(model)
    fuse_model.eval()
    fuse_out = fuse_model(inputs)

    diff = np.linalg.norm(ori_out.numpy() - fuse_out.numpy())
    print("single difference: ", diff)


def test_model():
    model = nn.Sequential(
        nn.ConvBatchNormReLU(3, 16, 7, 4, device=device, dtype=dtype),
        nn.ConvBatchNormReLU(16, 32, 3, 3, device=device, dtype=dtype),
    )
    inputs = ndl.Tensor(np.random.randn(1, 3, 32, 32), device=device)
    model.eval()
    ori_out = model(inputs)

    fuse_model = fuse_conv_bn_relu(model)
    fuse_model.eval()
    fuse_out = fuse_model(inputs)

    diff = np.linalg.norm(ori_out.numpy() - fuse_out.numpy())
    print("model difference: ", diff)


if __name__ == "__main__":

    test_single()
    test_model()
