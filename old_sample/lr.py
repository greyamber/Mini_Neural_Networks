from Base_Symbol import Symbol
import numpy as np
import ActivateFunctions as AF
from matplotlib import pyplot as plt

w1 = Symbol(name="Trainable", Trainable=True)

w1.set_value(np.random.random([1, 1]))

xs = Symbol()
ys = Symbol()

h = AF.MatDot(xs, w1)

loss = (ys - h) * (ys - h)

x = np.array(np.arange(-100, 100)).reshape([-1, 1]) / 100
y = x / 100


x_train = x[0:150]
y_train = y[0:150]

x_test = x[150:]
y_test = y[150:]


for i in range(100):
    xs.set_value(x_train)
    ys.set_value(y_train)
    loss.fp()
    print(w1.value)
    loss.bp()
    loss.gradient_decent(0.001)
    loss.clear_all()

xs.set_value(x_train)
ys.set_value(y_train)
loss.fp()
plt.scatter(x_train, y_train)
plt.scatter(x_train, h.value)
plt.show()
