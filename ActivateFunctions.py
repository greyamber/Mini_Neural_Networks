import Base_Function as bf
import Environment as env
import Base_Symbol
import numpy as np


def Sigmoid(x1):
    return Base_Symbol.Symbol(bf.BaseSigmoid(x1), name="Sigmoid")


def ReLu(x1):
    return Base_Symbol.Symbol(bf.BaseReLu(x1), name="ReLu")


def MatDot(x1, x2):
    return Base_Symbol.Symbol(bf.BaseMatDot(x1, x2), name="matdot")


def reduce_sum(x1, axis=None):
    return Base_Symbol.Symbol(bf.Base_reduce_sum(x1, axis=axis), name="reduce_sum")


def softmax(x1):
    return Base_Symbol.Symbol(bf.BaseSoftmax(x1), name="softmax")


def log(x1):
    return Base_Symbol.Symbol(bf.BaseLog(x1), name="Log")


if __name__ == "__main__":
    x1 = Base_Symbol.Symbol()
    x2 = Base_Symbol.Symbol()
    b2 = Base_Symbol.Symbol()
    b3 = Base_Symbol.Symbol()
    x3 = Base_Symbol.Symbol()
    x1.set_value(np.array([[1, 2]]))
    x2.set_value(np.array([[1,3],[4,5]]))
    x3.set_value(np.array([[2,1],[1,4]]))
    b2.set_value(np.ones([1,2]))
    b3.set_value(np.ones([1,2]))

    l1 = MatDot(x1, x2) + b2
    l2 = MatDot(l1, x3)
    l3 = reduce_sum(l2) / 10

    l3.fp()
    l3.bp()
    print(l2.value)
    print(l3.value)
    print(b2.gradient)
    print(x2.gradient)


