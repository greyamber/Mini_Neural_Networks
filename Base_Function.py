# coding: utf-8
from abc import ABCMeta, abstractmethod
import numpy as np


class Function(object):
    def __init__(self, func_type, x1=None, x2=None):
        self.func_type = func_type
        self.x1 = x1
        self.x2 = x2

    @abstractmethod
    def bp_method(self, y):
        pass

    @abstractmethod
    def fp_method(self, y):
        pass


class RootFunction(Function):
    def __init__(self, x1=None, x2=None):
        super(RootFunction, self).__init__("Root", x1, x2)
        self.value = None

    def fp_method(self):
        return self.value

    def bp_method(self, y):
        return None, None


class ConstantFunction(Function):
    def __init__(self, x1=None, x2=None):
        super(ConstantFunction, self).__init__("Constant", x1, x2)
        self.value = None

    def fp_method(self):
        return self.value

    def bp_method(self, y):
        return None, None


def mat_oper_vec(x1shape, x2shape, x1g, x2g):
    if x1shape != x1g.shape:
        if np.sum(x1g, axis=1).shape == x1shape:
            x1g = np.sum(x1g, axis=1)
        elif np.sum(x1g, axis=0).shape == x1shape:
            x1g = np.sum(x1g, axis=0)
        else:
            x1g = np.ones(x1shape) + x1g

    if x2shape != x2g.shape:
        if np.sum(x2g, axis=1, keepdims=True).shape == x2shape:
            x2g = np.sum(x2g, axis=1, keepdims=True)
        elif np.sum(x2g, axis=0, keepdims=True).shape == x2shape:
            x2g = np.sum(x2g, axis=0, keepdims=True)
        else:
            x2g = np.ones(x2shape) + x2g
    return x1g, x2g


class Addition(Function):
    def __init__(self, x1=None, x2=None):
        super(Addition, self).__init__("Add", x1, x2)

    def fp_method(self):
        return self.x1.value + self.x2.value

    def bp_method(self, y):
        # mat + mat or vec + vec or num + num
        x1g = x2g = y
        # mat + vec or vec + mat
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class Multiplication(Function):
    def __init__(self, x1=None, x2=None):
        super(Multiplication, self).__init__("Mul", x1, x2)

    def fp_method(self):
        return self.x1.value * self.x2.value

    def bp_method(self, y):
        x1g = self.x2.value * y
        x2g = self.x1.value * y
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class Subtraction(Function):
    def __init__(self, x1=None, x2=None):
        super(Subtraction, self).__init__("Sub", x1, x2)

    def fp_method(self):
        return self.x1.value - self.x2.value

    def bp_method(self, y):
        x1g = y
        x2g = -y
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class Division(Function):
    def __init__(self, x1=None, x2=None):
        super(Division, self).__init__("Div", x1, x2)

    def fp_method(self):
        return self.x1.value / self.x2.value

    def bp_method(self, y):
        x1g = 1.0 / self.x2.value * y
        x2g = y * self.x1.value * (-1.0) / self.x2.value / self.x2.value
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class _ReLu(Function):  # under building!
    def __init__(self, x1=None, x2=None):
        super(_ReLu, self).__init__("Relu", x1, x2)

    def fp_method(self):
        return np.where(self.x1.value > 0, self.x1.value, np.zeros(self.x1.value.shape))

    def bp_method(self, y):
        return np.where(self.x1.value > 0, np.zeros(self.x1.value.shape)+y, np.zeros(self.x1.value.shape)), None


class _Sigmoid(Function):
    def __init__(self, x1=None, x2=None):
        super(_Sigmoid, self).__init__("Sigmoid", x1, x2)

    def fp_method(self):
        return 1.0 / (1.0 + np.exp(-self.x1.value))

    def bp_method(self, y):
        x_ = self.fp_method()
        # d_sigmoid(x1) / d_x1 = (1-sigmoid(x1)) * sigmoid(x1)
        return y * (1 - x_) * x_, None


class _MatDot(Function):
    def __init__(self, x1=None, x2=None):
        super(_MatDot, self).__init__("MatDot", x1, x2)

    def fp_method(self):
        return np.array(np.dot(self.x1.value, self.x2.value))

    def bp_method(self, y):
        x1g = np.zeros(self.x1.value.shape)
        for i in range(self.x1.value.shape[0]):
            for j in range(self.x1.value.shape[1]):
                for m in range(self.x2.value.shape[1]):
                    x1g[i][j] += self.x2.value[j][m] * y[i][m]

        x2g = np.zeros(self.x2.value.shape)
        for j in range(self.x2.value.shape[1]):
            for i in range(self.x2.value.shape[0]):
                for n in range(self.x1.value.shape[0]):
                    x2g[i][j] += self.x1.value[n][i] * y[n][j]
        return x1g, x2g


if __name__ == "__main__":
    a = np.array([[-1,2,3],[0,1,-3],[6,3,1]])
    b = np.array([[1,2],[3,4],[5,6]])
    print(np.dot(a,b))



