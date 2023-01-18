import sys
from typing_extensions import NamedTuple

sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./tools")
sys.path.append("./utils")
import numpy as np
import pytest
import torch
import itertools

import needle as ndl
import needle.nn as nn

from simple_training import *
from utils import *


if __name__ == "__main__":
    print("testing model training")

    trainset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    valset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=False)
    train_loader = ndl.data.DataLoader(dataset=trainset, batch_size=128, shuffle=False)
    val_loader = ndl.data.DataLoader(dataset=valset, batch_size=1, shuffle=False)
    epochs = 1

    model1 = ResNet9(device=device, dtype="float32")

    out1 = train_cifar10(model1, train_loader, n_epochs=epochs, iter_limit=20)
    print("-------training cifar10 with resnet9------")
    print("avg acc: ", out1[0], " avg loss: ", out1[1])
    save_model(model1, "./ckpt/resnet9_model.pkl")
    print(
        "model1 params ", len(model1.named_parameters()), model1.parameters()[0].sum()
    )

    model2 = ResNet9(device=device, dtype="float32")
    print(
        "model2 params ", len(model2.named_parameters()), model2.parameters()[0].sum()
    )
    model2 = load_model(model2, "./ckpt/resnet9_model.pkl")
    # model2 = load_named_params(model2, model1.named_parameters())
    print(
        "model2 load params ",
        len(model2.named_parameters()),
        model2.parameters()[0].sum(),
    )

    # val_loader doesn't produce same 20 images, traverse whole dataset
    # to get same results between model1 and model2
    print("-------evaluating cifar10 with model1------")
    out2 = evaluate_cifar10(model1, val_loader, iter_limit=20)
    print("avg acc: ", out2[0], " avg loss: ", out2[1])

    print("-------evaluating cifar10 with model2------")
    out3 = evaluate_cifar10(model2, val_loader, iter_limit=20)
    print("avg acc: ", out3[0], " avg loss: ", out3[1])
