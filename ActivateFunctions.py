import Base_Function as bf
import Environment as env
import Base_Symbol
import numpy as np


def Sigmoid(x1):
    return Base_Symbol.Symbol(bf._Sigmoid(x1), name="Sigmoid")


def ReLu(x1):
    return Base_Symbol.Symbol(bf._ReLu(x1), name="ReLu")


def MatDot(x1, x2):
    return Base_Symbol.Symbol(bf._MatDot(x1, x2), name="matdot")


def reduce_sum(x1, x1shape, axis=None):
    axis1_feed = np.ones([x1shape[1], 1])
    axis1 = Base_Symbol.Symbol(name="reduce", keep_value=axis1_feed)
    axis0_feed = np.ones([1, x1shape[0]])
    axis0 = Base_Symbol.Symbol(name="reduce", keep_value=axis0_feed)

    if axis is None:
        ret = MatDot(axis0, x1)
        ret = MatDot(ret, axis1)
    elif axis == 0:
        ret = MatDot(axis0, x1)
    else:
        ret = MatDot(x1, axis1)
    return ret

def softmax(x1):
    return Base_Symbol.Symbol(bf._Softmax(x1), name="softmax")


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
    l3 = reduce_sum(l2, [1,2]) / 10

    l3.fp()
    l3.bp()
    print(l2.value)
    print(l3.value)
    print(b2.gradient)
    print(x2.gradient)


