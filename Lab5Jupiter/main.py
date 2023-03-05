import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


#       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates    alcohol  quality
# 1199            7.9             0.580         0.23            2.30      0.076                 23.0                  94.0  0.99686  3.21       0.58   9.500000        6

df = pd.read_csv('data/winequality-red.csv')
print(df.head(10))


# Check for multicollinearity
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('Correlation between columns')
sn.heatmap(df.corr(), ax=axis, annot=True)


# see how scatters fixed acidity and citric acidity,
# how citric acidity and ph, fixed acidity and ph,
# fixed acidity and density, volatile acidity and citric acidity
plot_rows = 3
plot_cols = 2
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
dependent_pairs = [
    ('fixed acidity', 'citric acid'),
    ('citric acid', 'pH'),
    ('fixed acidity', 'pH'),
    ('fixed acidity', 'density'),
    ('volatile acidity', 'citric acid'),
    ('volatile acidity', 'fixed acidity')
]
for index, pair in enumerate(dependent_pairs):
    ax = axis[index // plot_cols][index % plot_cols]
    np_x = df[pair[0]].to_numpy().reshape((-1, 1))
    np_y = df[pair[1]].to_numpy()
    model = LinearRegression().fit(np_x, np_y)
    predictions = model.predict(np_x)
    ax.set_xlabel(pair[0])
    ax.set_ylabel(pair[1])
    ax.set_title(f'{pair[0]} and {pair[1]}')
    ax.scatter(df[pair[0]], df[pair[1]], color='blue')
    ax.plot(np_x, predictions, color='red')

# drop "free sulfur dioxide", "pH", "residual sugar"
df = df.drop(["free sulfur dioxide", "pH", "residual sugar"], axis=1)

# graphs projection between factor and quality
plot_rows = 3
plot_cols = 3
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
for index, col_name in enumerate(df.columns):
    ax = axis[index // plot_cols][index % plot_cols]
    ax.set_title(f'{col_name} and quality')
    ax.scatter(df[col_name], df['quality'])

# Test and train data
df_np = df.to_numpy()
train_data_percentage = 0.75
train_size = int(df_np.shape[0] * train_data_percentage)
x_train, y_train = df_np[:train_size, :-1], df_np[:train_size, -1]
x_test, y_test = df_np[train_size:, :-1], df_np[train_size:, -1]

# Linear regression
linear_model = LinearRegression().fit(x_train, y_train)
linear_predictions_train = linear_model.predict(x_train)
linear_predictions_test = linear_model.predict(x_test)

tests = [
    ('Train', linear_predictions_train, y_train),
    ('Test', linear_predictions_test, y_test),
]
plot_rows = 1
plot_cols = 2
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
for index, t in enumerate(tests):
    print(f'Linear {t[0]} MAE: {mean_absolute_error(t[1], t[2])}')
    print(f'Linear {t[0]} MSE: {mean_squared_error(t[1], t[2])}')
    axis[index].set_xlabel(f'{t[0]} predictions')
    axis[index].set_ylabel(f'{t[0]} actual values')
    axis[index].scatter(t[1], t[2])
    axis[index].axline([0, 0], [1, 1], color='red')


# Polynomial regression
plot_rows = 4
plot_cols = 2
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
for index, degree in enumerate(range(1, plot_rows + 1)):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly_features.fit_transform(x_train)
    reg = LinearRegression()
    reg.fit(x_poly, y_train)
    x_train_vals = x_train.copy()
    x_test_vals = x_test.copy()
    x_train_vals_poly = poly_features.transform(x_train_vals)
    x_test_vals_poly = poly_features.transform(x_test_vals)
    polynomial_predictions_train = reg.predict(x_train_vals_poly)
    polynomial_predictions_test = reg.predict(x_test_vals_poly)
    tests = [
        ('Train', polynomial_predictions_train, y_train),
        ('Test', polynomial_predictions_test, y_test),
    ]
    for i, t in enumerate(tests):
        print(f'Polynomial {degree} {t[0]} MAE: {mean_absolute_error(t[1], t[2])}')
        print(f'Polynomial {degree} {t[0]} MSE: {mean_squared_error(t[1], t[2])}')
        axis[index][i].set_title(f'Polynomial {degree} {t[0]}')
        axis[index][i].set_xlabel(f'{t[0]} predictions')
        axis[index][i].set_ylabel(f'{t[0]} actual values')
        axis[index][i].scatter(t[1], t[2])
        axis[index][i].axline([0, 0], [1, 1], color='red')

plt.show()