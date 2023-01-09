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
  model1 = ResNet9(device=ndl.cpu())
  print('model1 params ', len(model1.named_parameters()), model1.parameters()[0])
  save_model(model1, './data/resnet9_model.pkl')
  model2 = ResNet9(device=ndl.cpu())
  print('model2 params ', len(model1.named_parameters()), model1.parameters()[0])
  model2 = load_model(model2, './data/resnet9_model.pkl')
  print('model2 load params ', len(model1.named_parameters()), model1.parameters()[0])



  
