import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
from sklearn.preprocessing import PolynomialFeatures

random.seed(42)
np.random.seed(42)

num_points = 50

# Generate x values
x = np.linspace(0, 1, num_points)

# Generate y values with noise
a = 1.5
b = 1

mean = 0
var = 1

epsilon = np.random.normal(mean, var, num_points)
y = a * x + b + epsilon

# Reshape x for sklearn
x_reshaped = x.reshape(-1, 1)

# features = 5
# poly_features = PolynomialFeatures(degree=5)
# x = poly_features.fit_transform(x_reshaped)

train_size = int(len(x)*0.8)
train_inds = np.random.choice(np.arange(len(x_reshaped)), size=train_size, replace=False)

train_mask = np.array([(i in train_inds) for i in range(len(x_reshaped))])
test_mask = ~train_mask


y_train = y[train_mask]
x_train = x_reshaped[train_mask]

y_test = y[test_mask]
x_test = x_reshaped[test_mask]

# Fit linear regression model
model = LinearRegression(fit_intercept=True)
model.fit(x_train, y_train)

# Predict y values
y_pred = model.predict(x_reshaped)

# Calculate residuals
residuals = y - y_pred

# Plotting

to_plot = 10

train_plots = np.random.choice(np.arange(len(x_train)), size=to_plot, replace=False)
test_plots = np.random.choice(np.arange(len(x_test)), size=to_plot, replace=False)

plt.figure(figsize=(10, 6))
plt.scatter(x_train[train_plots], y_train[train_plots], color='blue', label='Træningspunkter')
plt.scatter(x_test[[test_plots]], y_test[test_plots], color='purple', label='Testpunkter', marker='x')
plt.plot(x, y_pred, color='green', label=f'y_hat = {model.coef_} * x + {model.intercept_}')



# Plot residuals as red lines
for i, j in zip(train_plots, test_plots):
    pred_1 = model.predict(x_train[i].reshape(1,-1))
    pred_2 = model.predict(x_test[j].reshape(1,-1))
    plt.plot([x_train[i], x_train[i]], [y_train[i], pred_1[0]], color='red')
    plt.plot([x_test[j], x_test[j]], [y_test[j], pred_2[0]], color='red')


plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Lineær Regression, y_rigtig = x * {a} + {b} + N({mean},{var})')
# plt.text(0.5, -1, "Here goes text", horizontalalignment='center', verticalalignment='center')
plt.legend()
plt.grid(True)
plt.show()