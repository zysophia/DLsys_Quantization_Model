import sys
from typing_extensions import NamedTuple
sys.path.append('./python')
sys.path.append('./apps')
sys.path.append('./tools')
import numpy as np
import pytest
import torch
import itertools

import needle as ndl
import needle.nn as nn

from simple_training import *
from model_save_and_load import *


if __name__=="__main__":
  print("testing quantize")
  # model = ResNet9(device=ndl.cpu())
  # named_para = model.named_parameters()
  # print(len(named_para), len(model.parameters()))
  # for k, v in named_para.items():
  #   named_para[k] = nn.Parameter(nn.init.zeros(*v.shape))
  # load_named_params(model, named_para)
  # print(len(model.named_parameters()), len(model.parameters()))
  # print([x.sum() for x in model.parameters()])

  dataset1 = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
  dataloader1 = ndl.data.DataLoader(\
            dataset=dataset1,
            batch_size=128,
            shuffle=False
            # collate_fn=ndl.data.collate_ndarray,
            # drop_last=False,
            # device=device,
            # dtype="float32"
            )
  model1 = ResNet9(device=device, dtype="float32")
  epochs = 1
  out1 = train_cifar10(model1, dataloader1, n_epochs=epochs, iter_limit=20)
  print("-------training cifar10 with resnet9------")
  print("avg acc: ", out1[0], " avg loss: ", out1[1])

  print('model1 params ', len(model1.named_parameters()), model1.parameters()[0].sum())
  # save_model(model1, './data/resnet9_model.pkl')

  model2 = ResNet9(device=device, dtype="float32")
  print('model2 params ', len(model2.named_parameters()), model2.parameters()[0].sum())
  out2 = train_cifar10(model2, dataloader1, n_epochs=epochs, iter_limit=1)
  print("-------evaluating cifar10 with resnet9------")
  print("avg acc: ", out2[0], " avg loss: ", out2[1])

  # model2 = load_model(model2, './data/resnet9_model.pkl')
  model2 = load_named_params(model2, model1.named_parameters())
  print('model2 load params ', len(model2.named_parameters()), model2.parameters()[0].sum())
  out3 = train_cifar10(model2, dataloader1, n_epochs=epochs, iter_limit=1)
  print("-------evaluating cifar10 with resnet9------")
  print("avg acc: ", out3[0], " avg loss: ", out3[1])

  out2 = evaluate_cifar10(model1, dataloader1)
  # print("-------training cifar10 with resnet9------")
  # print("avg acc: ", out2[0], " avg loss: ", out2[1])


  
