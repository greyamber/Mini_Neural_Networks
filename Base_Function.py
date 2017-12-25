# coding: utf-8
from abc import ABCMeta, abstractmethod
import numpy as np


class Function(object):
    """
    func_type：操作名，注意Root，Constant等有特殊用途，继承时不要使用已有的type
    何时需要自定义新的fp与bp：
    1.认为该操作用已有的加减乘除relu和sigmoid等难以实现
    2.认为用已有方法实现后性能很差
    3.编写独有的cuda计算
    4.认为最好在确定之前的操作的具体值后才计算该操作的维度等必要信息（比如reduce_sum等）
    如何自定义新的fp与bp操作：
    1.继承Function类，定义一个独有的func_type字符串
    2.只支持二元或一元操作，左数为x1，右数位x2；x1，x2为Symbol类
    3.重写fp_method(self)：前向传播的函数，使用x1.value与x2.value，返回一个值（np.array）
    4.重写bp_method(self, y)：反向传播的函数，y为上一层的梯度。返回x1g和x2g两个梯度
      x2未使用则x2g为None
      例子：out=g(f(x)), out对x求导：y = dout/df(x) --> dout/dx = y*df(x)/d(x)
    5.注意期间尽量不要改变维度，即：keep_dim
    """
    def __init__(self, func_type, x1=None, x2=None):
        self.func_type = func_type
        self.x1 = x1
        self.x2 = x2

    @abstractmethod
    def bp_method(self, y):
        pass

    @abstractmethod
    def fp_method(self):
        pass


class RootFunction(Function):
    def __init__(self, x1=None, x2=None):
        super(RootFunction, self).__init__("Root", None, None)
        self.value = None

    def fp_method(self):
        return self.value

    def bp_method(self, y):
        return None, None


class ConstantFunction(Function):
    def __init__(self, x1=None, x2=None):
        super(ConstantFunction, self).__init__("Constant", None, None)
        self.value = None

    def fp_method(self):
        return self.value

    def bp_method(self, y):
        return None, None


def mat_oper_vec(x1shape, x2shape, x1g, x2g):
    """
    实现向量+矩阵，数+向量，数+矩阵的倒数维度变换
    :param x1shape: 第一个参数的shape
    :param x2shape: 第二个参数的shape
    :param x1g: 第一个参数（梯度）
    :param x2g: 第二个参数（梯度）
    :return: 修饰后的梯度（x1g，x2g）
    """
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
    def __init__(self, x1, x2):
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
    def __init__(self, x1, x2):
        super(Multiplication, self).__init__("Mul", x1, x2)

    def fp_method(self):
        return self.x1.value * self.x2.value

    def bp_method(self, y):
        x1g = self.x2.value * y
        x2g = self.x1.value * y
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class Subtraction(Function):
    def __init__(self, x1, x2):
        super(Subtraction, self).__init__("Sub", x1, x2)

    def fp_method(self):
        return self.x1.value - self.x2.value

    def bp_method(self, y):
        x1g = y
        x2g = -y
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class Division(Function):
    def __init__(self, x1, x2):
        super(Division, self).__init__("Div", x1, x2)

    def fp_method(self):
        return self.x1.value / self.x2.value

    def bp_method(self, y):
        x1g = 1.0 / self.x2.value * y
        x2g = y * self.x1.value * (-1.0) / self.x2.value / self.x2.value
        x1g, x2g = mat_oper_vec(self.x1.value.shape, self.x2.value.shape, x1g, x2g)
        return x1g, x2g


class BaseReLu(Function):
    def __init__(self, x1, x2=None):
        super(BaseReLu, self).__init__("Relu", x1, x2)

    def fp_method(self):
        return np.where(self.x1.value > 0, self.x1.value, np.zeros(self.x1.value.shape))

    def bp_method(self, y):
        return np.where(self.x1.value > 0, np.zeros(self.x1.value.shape)+y, np.zeros(self.x1.value.shape)), None


class BaseSigmoid(Function):
    def __init__(self, x1, x2=None):
        super(BaseSigmoid, self).__init__("Sigmoid", x1, x2)

    def fp_method(self):
        return 1.0 / (1.0 + np.exp(-self.x1.value))

    def bp_method(self, y):
        x_ = self.fp_method()
        # d_sigmoid(x1) / d_x1 = (1-sigmoid(x1)) * sigmoid(x1)
        return y * (1 - x_) * x_, None


class BaseMatDot(Function):
    def __init__(self, x1, x2):
        super(BaseMatDot, self).__init__("MatDot", x1, x2)

    def fp_method(self):
        return np.array(np.dot(self.x1.value, self.x2.value))

    def bp_method(self, y):
        x1g = np.dot(y, np.matrix(self.x2.value).T)
        # equal to :
        # x1g = np.zeros(self.x1.value.shape)
        # for i in range(self.x1.value.shape[0]):
        #    for j in range(self.x1.value.shape[1]):
        #        for m in range(self.x2.value.shape[1]):
        #            x1g[i][j] += self.x2.value[j][m] * y[i][m]

        x2g = np.dot(np.matrix(self.x1.value).T, y)
        # equal to :
        # x2g = np.zeros(self.x2.value.shape)
        # for j in range(self.x2.value.shape[1]):
        #     for i in range(self.x2.value.shape[0]):
        #         for n in range(self.x1.value.shape[0]):
        #             x2g[i][j] += self.x1.value[n][i] * y[n][j]

        return np.array(x1g), np.array(x2g)


class BaseSoftmax(Function):
    def __init__(self, x1, x2=None):
        super(BaseSoftmax, self).__init__("Softmax", x1, x2)

    def fp_method(self):
        exp = np.exp(self.x1.value)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        return exp / exp_sum

    def bp_method(self, y):
        softmax = self.fp_method()
        x1g = softmax * (1 - softmax) * y
        return x1g, None


class Base_reduce_sum(Function):
    def __init__(self, x1, axis=None):
        super(Base_reduce_sum, self).__init__("reduce_sum", x1, None)
        self.axis = axis

    def fp_method(self):
        if self.axis is not None:
            ret = np.sum(self.x1.value, axis=self.axis, keepdims=True)
        else:
            ret = np.sum(self.x1.value, keepdims=True)
        return ret

    def bp_method(self, y):
        x1g = np.ones(self.x1.value.shape, np.float32) * y
        return x1g, None


class BaseLog(Function):
    def __init__(self, x1, x2=None):
        super(BaseLog, self).__init__("Ln", x1, x2)

    def fp_method(self):
        return np.log(self.x1.value)

    def bp_method(self, y):
        x1g = y / self.x1.value
        return x1g, None


if __name__ == "__main__":
    a = np.array([[-1,2,3],[0,1,-3],[6,3,1]])
    b = np.array([[1,1,3]])
    print(a * b)



