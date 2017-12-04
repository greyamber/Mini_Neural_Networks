from symbol import Symbol
import numpy as np
import ActivateFunctions as AF
from matplotlib import pyplot as plt


w1 = Symbol(name="Trainable", Trainable=True)
b1 = Symbol(name="Trainable", Trainable=True)

w2 = Symbol(Trainable=True, name="Trainable")
b2 = Symbol(Trainable=True, name="Trainable")

w3 = Symbol(name="Trainable", Trainable=True)
b3 = Symbol(name="Trainable", Trainable=True)

w1.set_value(np.random.random([1, 10]))
b1.set_value(np.zeros([150, 10]))

w3.set_value(np.random.random([10, 10]))
b3.set_value(np.zeros([150, 10]))

w2.set_value(np.random.random([10,1]))
b2.set_value(np.zeros([150,1]))

xs = Symbol()
ys = Symbol()

l1 = AF.ReLu(AF.MatDot(xs, w1) + b1)
l2 = AF.ReLu(AF.MatDot(l1, w3) + b3)
h = AF.MatDot(l2, w2) + b2

loss = (ys - h) * (ys - h)


#x = np.array(list(zip(np.arange(-100,100), np.arange(-100,100))))
#y = np.sum(x, axis=0) * 0.01
x = np.arange(-100, 100).reshape([-1, 1])
y = x*x / 1000 + 0.3*x + 0.001*x*x*x

x_train = x[0:150]
y_train = y[0:150]

x_test = x[150:]
y_test = y[150:]


for i in range(100):
    xs.set_value(x_train)
    ys.set_value(y_train)
    loss.fp()
    print(w2.value)
    loss.bp()
    loss.gradient_decent(0.01)
    loss.clear_all()

xs.set_value(x_train)
ys.set_value(y_train)
loss.fp()
plt.plot(x_train, y_train)
plt.plot(x_train, h.value)
plt.show()








