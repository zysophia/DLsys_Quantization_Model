"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        # print("before EWiseMul", a.dtype, b.dtype)
        # print("after  EWiseMul", (a*b).dtype)
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        # print("before MulScalar", a.dtype)
        # print("after  MulScalar", (a * self.scalar).dtype)
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # print("before EWiseDiv", a.dtype, b.dtype)
        # print("after  EWiseDiv", (a /b).dtype)
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return rhs**(-1) * out_grad, -lhs * rhs**(-2) * out_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print("before DivScalar", a.dtype)
        # print("after  DivScalar", (array_api.divide(a, self.scalar, dtype=a.dtype)).dtype)
        return array_api.divide(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ret = array_api.swapaxes(a, *self.axes if self.axes else (-1, -2))
        # print("before Transpose", a.dtype)
        # print("after  Transpose", ret.dtype)
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, reversed(self.axes) if self.axes else (-1, -2))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.ori_shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.ori_shape = a.shape
        ret = a.reshape(self.shape)
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, self.ori_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.ori_shape = shape

    def compute(self, a):
        self.ori_shape = a.shape
        ret = array_api.broadcast_to(a, self.shape)
        # print("before BroadcastTo", a.dtype)
        # print("after  BroadcastTo", ret.dtype)
        return ret

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape1 = [1]*(len(self.shape)-len(self.ori_shape)) + list(self.ori_shape)
        shape2 = list(self.shape)
        # print(shape1, shape2)
        axis = tuple(i for i in range(len(shape2)) if shape1[i] == 1)
        # print(axis)
        return reshape(summation(out_grad, axes = axis), self.ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.ori_shape = None
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.ori_shape = a.shape
        ret = array_api.sum(a, axis = self.axes, dtype=a.dtype)
        # print("before Summation", a.dtype)
        # print("after  Summation", ret.dtype)
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape1 = list(self.ori_shape)
        # print(self.ori_shape)
        axes = self.axes if self.axes else tuple(range(len(self.ori_shape)))
        # print(axes)
        for x in list(axes):
            shape1[x] = 1
        return broadcast_to(reshape(out_grad, shape1), self.ori_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # print("before MatMul", a.dtype, b.dtype)
        # print("after  MatMul", (a@b).dtype)
        # print("aaabbb", a.shape, b.shape)
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # print(lhs.dtype, rhs.dtype)
        lsize, rsize = len(lhs.shape), len(rhs.shape)
        if 2 <= lsize < rsize:
            left = summation(out_grad @ rhs.transpose(), axes = tuple(range(rsize-lsize)))
        else:
            left = out_grad @ rhs.transpose()
        if 2 <= rsize < lsize:
            right = summation(lhs.transpose() @ out_grad, axes = tuple(range(lsize-rsize)))
        else:
            right = lhs.transpose() @ out_grad
        # print(left.dtype, right.dtype)
        return left, right
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node)**(-1) * out_grad
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        array = node.realize_cached_data()
        tensor = Tensor(array_api.greater(array, 0), dtype="float32")
        return out_grad * tensor
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # c = array_api.max(Z)
            # return array_api.log(array_api.sum(array_api.exp(Z - c))) + c
            self.axes = tuple(range(len(Z.shape)))
        cc = array_api.max(Z, axis=self.axes)
        # print(self.axes)
        # print(list(self.axes), range(len(Z.shape)))
        left_axes = [1 if i in list(self.axes) else Z.shape[i] for i in range(len(Z.shape))]
        c = array_api.reshape(cc, tuple(left_axes))
        c = array_api.broadcast_to(c, Z.shape)
        return array_api.log(array_api.sum(array_api.exp(Z - c), axis = self.axes)) + cc
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        # print('Z: ', Z)
        cc = array_api.max(Z.realize_cached_data(), axis=self.axes)
        cc = Tensor(cc, dtype="float32")
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        left_axes = [1 if i in list(self.axes) else Z.shape[i] for i in range(len(Z.shape))]
        c = cc.reshape(tuple(left_axes)).broadcast_to(Z.shape)
        # print("c broadcast: ", c)
        expo = (Z - c).exp()
        # print("expo: ", expo)
        sumexp = expo.sum(axes = self.axes)
        sumexp = sumexp.reshape(tuple(left_axes)).broadcast_to(Z.shape)
        # print("sumexp: ", sumexp)
        out_grad = out_grad.reshape(tuple(left_axes)).broadcast_to(Z.shape)
        # print("out_grad: ", out_grad)
        return out_grad*expo/sumexp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
