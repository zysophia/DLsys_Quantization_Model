"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.usebias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=None, dtype="float32", requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=None, dtype="float32", requires_grad=True).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        matmul = X @ self.weight
        if self.usebias:
            matmul += self.bias.broadcast_to(matmul.shape)
            # print(matmul)
        return matmul
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        reshape = 1
        for s in X.shape[1:]:
            reshape *= s
        return X.reshape((X.shape[0], reshape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for mod in self.modules:
            y = mod(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_onehot = init.one_hot(logits.shape[1], y)
        # print("yonehot: ", y_onehot)
        logsum = ops.logsumexp(logits, axes = (1,))
        # print("logsum: ", logsum)
        return (logsum.sum()-(logits*y_onehot).sum())/logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        mean = (x.sum(axes=(0,))/N).reshape((1,-1))
        var = (((x-mean.broadcast_to(x.shape))**2).sum(axes=(0,))/N).reshape((1,-1))
        if self.training:
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean.reshape(-1)
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var.reshape(-1)
            return (x-mean.broadcast_to(x.shape))/(var.broadcast_to(x.shape) +self.eps)**0.5 * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        else:
            return (x-self.running_mean.broadcast_to(x.shape))/(self.running_var.broadcast_to(x.shape) +self.eps)**0.5 * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[1]
        # print(x)
        mean = (x.sum(axes=(1,))/N).reshape((-1,1))
        # print("mean: ", mean)
        var = (((x-mean.broadcast_to(x.shape))**2).sum(axes=(1,))/N).reshape((-1,1))
        # print("var: ", var)
        return (x-mean.broadcast_to(x.shape))/(var.broadcast_to(x.shape) +self.eps)**0.5 * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            probs = init.randb(*x.shape, p = 1-self.p, device=x.device)
            return x * probs/(1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



