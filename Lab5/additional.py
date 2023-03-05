import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


#                                         COUNTRY  ISO                                        UA          Cql           Ie          Iec           Is
# 0                                       Albania  ALB                                   Албанія   0,97392353  0,605347614  0,538672856  0,510112666
# 1                                       Algeria  DZA                                     Алжир  0,782134498   0,58721932  0,348159396  0,497985576
# 2                                        Angola  AGO                                    Ангола  0,372343539   0,27439361  0,332117384  0,346906645
# 3                                     Argentina  ARG                                 Аргентина  0,883830062  0,699685109   0,28199471  0,518820368


def read_dataset(path: str, sep: str = ';') -> pd.DataFrame:
    data = pd.read_csv(path, sep=sep)
    return data


def replace_comma_with_dots(dataset: pd.DataFrame, column_name: str) -> None:
    dataset[column_name] = dataset[column_name].astype(str)
    dataset[column_name] = dataset[column_name].str.replace(',', '.')


def convert_column_to_float(dataset: pd.DataFrame, column_name: str) -> None:
    dataset[column_name] = dataset[column_name].astype(float)


def replace_nan_with_mean(dataset: pd.DataFrame, column_name: str):
    mean_value = dataset[column_name].mean()
    dataset[column_name].fillna(value=mean_value, inplace=True)


def convert_float_with_positive(dataset: pd.DataFrame, column_name: str):
    dataset[column_name] = dataset[column_name].abs()


def polynomial_regression(x: pd.DataFrame, y: pd.Series, degree: float):
    np_x = x.values
    np_y = y.to_numpy()
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly_features.fit_transform(np_x)
    reg = LinearRegression()
    reg.fit(x_poly, np_y)
    x_vals_poly = poly_features.transform(np_x)
    predictions = reg.predict(x_vals_poly)
    return np_x, np_y, predictions, poly_features, reg


dataset = pd.read_csv('data/Data4.csv', sep=';', encoding='cp1251')
dataset_test = pd.read_csv('data/Data4t.csv', sep=';', encoding='cp1251')
dataset.columns.values[0] = 'COUNTRY'
dataset_test.columns.values[0] = 'COUNTRY'

for d in [dataset, dataset_test]:
    for column_name in d.columns[3:]:
        replace_comma_with_dots(d, column_name)
        convert_column_to_float(d, column_name)
        replace_nan_with_mean(d, column_name)
        convert_float_with_positive(d, column_name)

df = dataset.loc[:, 'Cql':].copy()
df_test = dataset_test.loc[:, 'Cql':].copy()

# Check for multicollinearity
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('Correlation between columns')
sn.heatmap(df.corr(), ax=axis, annot=True)

# see how scatters fixed acidity and citric acidity,
# how citric acidity and ph, fixed acidity and ph,
# fixed acidity and density, volatile acidity and citric acidity
plot_rows = 2
plot_cols = 3
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
dependent_pairs = [
    ('Cql', 'Ie'),
    ('Cql', 'Iec'),
    ('Cql', 'Is'),
    ('Ie', 'Iec'),
    ('Ie', 'Is'),
    ('Iec', 'Is'),
]
for index, pair in enumerate(dependent_pairs):
    ax = axis[index // plot_cols][index % plot_cols]
    ax.scatter(df[pair[0]], df[pair[1]], color='blue')
    np_x, np_y, predictions, _, _ = polynomial_regression(df.loc[:, pair[0]].to_frame(), df[pair[1]], 1)
    ax.set_xlabel(pair[0])
    ax.set_ylabel(pair[1])
    ax.set_title(f'{pair[0]} and {pair[1]}')
    ax.plot(np_x, predictions, color='r')


# Regressions
plot_rows = 4
plot_cols = 2
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
np_test_x = df_test.iloc[:, :-1].to_numpy()
np_test_y = df_test.iloc[:, -1].to_numpy()
mses = []
for index, degree in enumerate(range(1, plot_rows + 1)):
    res = polynomial_regression(df.iloc[:, :-1], df.iloc[:, -1], 1)
    np_train_x, np_train_y, predictions_train, poly_features, model = res
    x_test_vals_poly = poly_features.transform(np_test_x)
    predictions_test = model.predict(x_test_vals_poly)
    tests = [
        ('Test', np_test_y, predictions_test),
        ('Train', np_train_y, predictions_train)
    ]
    for i, t in enumerate(tests):
        mae = mean_absolute_error(t[1], t[2])
        mse = mean_squared_error(t[1], t[2])
        if t[0] == 'Test':
            mses.append(mse)
        print(f'Polynomial {degree} {t[0]} MAE: {mae}')
        print(f'Polynomial {degree} {t[0]} MSE: {mse}')
        axis[index][i].set_title(f'Polynomial {degree} {t[0]}')
        axis[index][i].set_xlabel(f'{t[0]} predictions')
        axis[index][i].set_ylabel(f'{t[0]} actual values')
        axis[index][i].scatter(t[1], t[2])
        axis[index][i].axline([0, 0], [1, 1], color='red')

min_mse = min(mses)
degree = mses.index(min_mse) + 1
print(f'Polynomial {degree} Test MSE: {min_mse}')


plt.show()






