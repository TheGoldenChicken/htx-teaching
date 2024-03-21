import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

xs = np.linspace(1, 40)

a = 3
b = 2

train_indices = np.random.randint(0, 40, 30)

func = lambda x: x * a + b + np.random.randn()

ys = np.array(map(func, xs))

linear_rls = LinearRegression()
ridge = Ridge()

linear_rls.fit(xs[train_indices], ys[train_indices])

plt.plot(xs, linear_rls.predict(xs[train_indices]))