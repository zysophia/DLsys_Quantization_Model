"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # print(len(self.params))
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            pgrad = p.grad.realize_cached_data() + self.weight_decay * p.realize_cached_data()
            if idx not in self.u:
                self.u[idx] = pgrad * (1-self.momentum)
            else:
                self.u[idx] = self.u[idx]*self.momentum + pgrad * (1-self.momentum)
            self.params[idx].data = p - self.u[idx] * self.lr
            # break
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            pgrad = p.grad.realize_cached_data() + self.weight_decay * p.realize_cached_data()
            # m and v
            if idx not in self.m:
                self.m[idx] = pgrad * (1-self.beta1)
            else:
                self.m[idx] = self.m[idx]*self.beta1 + pgrad * (1-self.beta1)
            if idx not in self.v:
                self.v[idx] = pgrad*pgrad * (1-self.beta2)
            else:
                self.v[idx] = self.v[idx]*self.beta2 + pgrad*pgrad * (1-self.beta2)
            mhat = self.m[idx]/(1-self.beta1**self.t)
            vhat = self.v[idx]/(1-self.beta2**self.t)
            self.params[idx].data = p.data - mhat/ (vhat**0.5 + self.eps) * self.lr
        ### END YOUR SOLUTION
