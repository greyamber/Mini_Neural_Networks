import Base_Function as bf
import symbol
import numpy as np


def Sigmoid(x1):
    return symbol.Symbol(bf._Sigmoid(x1), name="Sigmoid")


def ReLu(x1):
    return symbol.Symbol(bf._ReLu(x1), name="ReLu")


def MatDot(x1, x2):
    return symbol.Symbol(bf._MatDot(x1, x2), name="matdot")


if __name__ == "__main__":
    x1 = symbol.Symbol(name="??")
    x2 = symbol.Symbol()
    x3 = symbol.Symbol()

    y = x1 / x2 + x2 * (x3 + x1) / 10 - x1

    x1.set_value(np.array([1, 2, 3]))
    x2.set_value(np.array([2, 3, 4]))
    x3.set_value(np.array([3, 4, 5]))

    y = ReLu(y)

    y.fp()
    y.bp()
    print(y.value)
    print(y.gradient)
    print(x1.gradient)
    print(x2.gradient)
    print(x3.gradient)

    x4 = symbol.Symbol()
    x5 = symbol.Symbol()

    y2 = MatDot(x4, x5)

    x4.set_value(np.array([[1, 1], [1, 2]]))
    x5.set_value(np.array([[1, 2, 3], [1, 5, 6]]))
    y2.fp()
    y2.bp()
    print(y2.value)
    print(y2.gradient)
    print(x5.gradient)