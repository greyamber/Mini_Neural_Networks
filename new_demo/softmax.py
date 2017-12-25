import Base_Symbol as BS
import ActivateFunctions as AF
import layers
import numpy as np
import sklearn.datasets as skdata

X, y = skdata.load_iris(return_X_y=True)
X = X / np.max(X, axis=0)
X = X-np.mean(X, axis=0)
y = np.array([[0 if i!=j else 1 for j in range(3)] for i in y])


xs = BS.Placeholder()
ys = BS.Placeholder()

l1 = layers.dense_layer(xs, init_w=(np.random.random([4, 64]) - 0.5),
                        init_b=np.zeros([1, 64])+0.1, activation_func=AF.ReLu)

l2 = layers.dense_layer(l1, init_w=(np.random.random([64, 64]) - 0.5),
                        init_b=np.zeros([1, 64])+0.1, activation_func=AF.ReLu)

l3 = layers.dense_layer(l2, init_w=(np.random.random([64, 64]) - 0.5),
                        init_b=np.zeros([1, 64])+0.1, activation_func=AF.ReLu)

out = layers.dense_layer(l3, init_w=(np.random.random([64, 3]) - 0.5),
                         init_b=np.zeros([1, 3])+0.1, activation_func=AF.softmax)

loss = -1 * AF.reduce_sum(AF.log(out + 1e-6) * ys) / 32

print("Start")
for i in range(10000):
    sits = np.random.randint(0, 100, 32)
    xs.feed(X[sits])
    ys.feed(y[sits])
    loss.fp()
    loss.bp()

    lr = 1e-4
    print(loss.value)
    print(np.sum(out.value*ys.value)/32)
    loss.gradient_decent(lr)
    loss.clear_all()

