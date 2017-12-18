from Base_Symbol import Symbol
import numpy as np
import ActivateFunctions as AF
from matplotlib import pyplot as plt


w1 = Symbol(name="Trainable", Trainable=True)
b1 = Symbol(name="Trainable", Trainable=True)

w2 = Symbol(Trainable=True, name="Trainable")
b2 = Symbol(Trainable=True, name="Trainable")

w3 = Symbol(name="Trainable", Trainable=True)
b3 = Symbol(name="Trainable", Trainable=True)

w4 = Symbol(name="Trainable", Trainable=True)
b4 = Symbol(name="Trainable", Trainable=True)

w1.set_value(np.random.random([1, 10]) - 0.5)
b1.set_value(np.zeros([1, 10]) + 0.01)

w3.set_value(np.random.random([10, 10]) - 0.5)
b3.set_value(np.zeros([1, 10]) + 0.01)

w4.set_value(np.random.random([10, 10]) - 0.5)
b4.set_value(np.zeros([1, 10]) + 0.01)

w2.set_value(np.random.random([10, 1]) - 0.5)
b2.set_value(np.zeros([1, 1]) + 0.01)

xs = Symbol()
ys = Symbol()
t1 = AF.MatDot(xs, w1) + b1
l1 = AF.ReLu(t1)
l2 = AF.ReLu(AF.MatDot(l1, w3) + b3)
l3 = AF.ReLu(AF.MatDot(l2, w4) + b4)
h = AF.MatDot(l3, w2) + b2

loss = AF.reduce_sum((ys - h) * (ys - h), [150, 1]) / 150.0

x = np.arange(-100, 100).reshape([-1, 1]) / 100
y = 2*x*x

x_train = x[0:150]
y_train = y[0:150]

x_test = x[150:]
y_test = y[150:]


for i in range(10000):
    xs.set_value(x_train)
    ys.set_value(y_train)
    loss.fp()
    loss.bp()
    lr = 0.01
    print(loss.value)
    loss.gradient_decent(lr)
    loss.clear_all()

xs.set_value(x_train)
ys.set_value(y_train)
loss.fp()
plt.plot(x_train, y_train)
plt.plot(x_train, h.value)
plt.show()








