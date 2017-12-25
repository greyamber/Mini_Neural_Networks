import Base_Symbol as BS
import ActivateFunctions as AF
import numpy as np


def dense_layer(inputs, init_w, init_b=None, activation_func=None):
    W = BS.Variable(init=init_w, name="", Trainable=True)
    if init_b is not None:
        b = BS.Variable(init=init_b, name="", Trainable=True)
        out = AF.MatDot(inputs, W) + b
    else:
        out = AF.MatDot(inputs, W)

    if activation_func is not None:
        out = activation_func(out)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xs = BS.Placeholder()
    ys = BS.Placeholder()
    l1 = dense_layer(xs, init_w=np.random.random([1, 1]) - 0.5,
                     init_b=np.zeros([1, 1]) + 0.01, activation_func=AF.Sigmoid)
    l1 = dense_layer(l1, init_w=np.random.random([1, 1]) - 0.5,
                     init_b=np.zeros([1, 1]) + 0.01, activation_func=AF.Sigmoid)
    l1 = dense_layer(l1, init_w=np.random.random([1, 1]) - 0.5,
                     init_b=np.zeros([1, 1]) + 0.01)

    loss = AF.reduce_sum((l1 - ys) * (l1 - ys))

    for i in range(5000):
        xs.feed(np.arange(-100, 100).reshape([-1, 1]) / 100 - 0.5)
        ys.feed(np.arange(-100, 100).reshape([-1, 1]) / 10 + 9)
        loss.fp()
        loss.bp()
        lr = 0.0001
        print(loss.value)
        loss.gradient_decent(lr)
        loss.clear_all()

    xs.feed(np.arange(-100, 100).reshape([-1, 1]) / 100 - 0.5)
    ys.feed(np.arange(-100, 100).reshape([-1, 1]) / 10 + 9)
    loss.fp()
    plt.plot(np.arange(-100, 100).reshape([-1, 1]) / 100 - 0.5, np.arange(-100, 100).reshape([-1, 1]) / 10 + 9)
    plt.plot(xs.value, l1.value)
    plt.show()

