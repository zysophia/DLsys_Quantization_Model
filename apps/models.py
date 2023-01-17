import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import math
import numpy as np

np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.block1 = nn.ConvBatchNormReLU(3, 16, 7, 4, device=device, dtype=dtype)
        self.block2 = nn.ConvBatchNormReLU(16, 32, 3, 2, device=device, dtype=dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                nn.ConvBatchNormReLU(32, 32, 3, 1, device=device, dtype=dtype),
                nn.ConvBatchNormReLU(32, 32, 3, 1, device=device, dtype=dtype),
            )
        )
        self.block3 = nn.ConvBatchNormReLU(32, 64, 3, 2, device=device, dtype=dtype)
        self.block4 = nn.ConvBatchNormReLU(64, 128, 3, 2, device=device, dtype=dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                nn.ConvBatchNormReLU(128, 128, 3, 1, device=device, dtype=dtype),
                nn.ConvBatchNormReLU(128, 128, 3, 1, device=device, dtype=dtype),
            )
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(dataset[1][0].shape)
