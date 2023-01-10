import os
from typing_extensions import ParamSpecKwargs
import needle as ndl
import needle.nn as nn
from needle.autograd import Tensor
from collections.abc import Mapping
import pickle
import numpy as np


def save_model(model: nn.Module, file_path: str):
    named_paras = model.named_parameters()
    for k, v in named_paras.items():
        named_paras[k] = v.numpy()
    with open (file_path, 'wb') as f:
        pickle.dump(named_paras, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_model(model: nn.Module, file_path: str):
    with open (file_path, 'rb') as f:
        named_paras = pickle.load(f)
    f.close()
    for k, v in named_paras.items():
        named_paras[k] = nn.Parameter(v)
    return load_named_params(model, named_paras)


def load_named_params(model: nn.Module, named_params: dict):
    for k, v in named_params.items():
        state_dict = load_named_param(model, k, v)
    model.__dict__ = state_dict.__dict__
    return model


def load_named_param(model: nn.Module, name, param):
    state_dict = model.__dict__
    # print("-------name------: ", name)
    # print('-------dict------: ', state_dict)
    state_dict = update_param(name, param, state_dict)
    model.__dict__ = state_dict
    return model


def update_param(name: str, param: Tensor, d: object):
    keys = name.split('.', 1)
    curk = keys[0]
    if curk.isdigit():
        curk = int(curk) 
    if isinstance(d, nn.Module):
        return load_named_param(d, name, param)
    elif isinstance(d, tuple):
        dlist = list(d)
        if len(keys) > 1:
            subd = dlist[curk]
            dlist[curk] = update_param(keys[1], param, subd)
        else:
            dlist[curk] = param
        d = tuple(dlist)
    else:
        if len(keys) > 1:
            subd = d[curk]
            d[curk] = update_param(keys[1], param, subd)
        else:
            d[curk] = param
    return d

