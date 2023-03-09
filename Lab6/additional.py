import math

import pandas as pd
import numpy as np
# Візуалізація
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
# Табличний вивід
from tabulate import tabulate
# Визначення точки локтя (кластеризація)
from kneed import KneeLocator
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


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


data_path = 'data/Data2.csv'
df = read_dataset(data_path, sep=';')

df['Population'] = df['Populatiion']
df = df.drop(['Populatiion'], axis=1)

numeric_cols = df.columns[2:]

for column_name in numeric_cols:
    replace_comma_with_dots(df, column_name)
    convert_column_to_float(df, column_name)
    replace_nan_with_mean(df, column_name)
    convert_float_with_positive(df, column_name)
    print(column_name, df[column_name].dtype)

## density column
df['GDP'] = df['GDP per capita'] * df['Population']

df['Density'] = df['Population'] / df['Area']

print(df.head(10).to_string())

## kmeans
factors = df.columns.values.tolist()
factors = [f for f in factors if f not in ['Region', 'Country Name']]

kmeans_kwargs = {
    'init': 'random',
    'n_init': 10,
    'max_iter': 300,
    'random_state': 0,
}
sse = []
max_kernels = 10
features = df[factors]
for k in range(1, max_kernels + 1):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, max_kernels + 1), sse)
plt.xticks(range(1, max_kernels + 1))
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.grid(linestyle='--')


for column_name in factors:
    fig = sns.FacetGrid(df, aspect=4)
    fig.map(sns.kdeplot, column_name, fill=True)
    oldest = df[column_name].max()
    fig.set(xlim=(0, oldest))
    fig.add_legend()

df_nums = df[factors]
fig, axis = plt.subplots(figsize=(8, 6))
sns.heatmap(df_nums.corr(numeric_only=True), ax=axis, annot=True)


def is_linear(series_one: pd.Series, series_two: pd.Series) -> bool:
    return math.fabs(series_one.corr(series_two)) >= 0.8


for c_one in factors:
    corrs_row = [{c: is_linear(df[c_one], df[c])} for c in factors]
    pairs = {}
    for cc in corrs_row:
        pairs.update(cc)
    row = {c_one: pairs}
    print(row)


# Elbow point
kl = KneeLocator(range(1, max_kernels + 1), sse, curve='convex', direction='decreasing')
n_clusters_list = [kl.elbow, 3, 4, 5]

for n in n_clusters_list:
    kmeans = KMeans(
        init='random',
        n_clusters=n,
        n_init=10,
        max_iter=300,
        random_state=0
    )
    kmeans.fit(features)
    name = f'N-Clusters {n}'
    df[name] = kmeans.labels_

    print(name)
    for i in range(n):
        t = df[df[name] == i]
        table = pd.pivot_table(
            t, values=['GDP', 'Population', 'Area'], index=['Region'], aggfunc=np.sum
        )
        table['Density'] = table['Population'] / table['Area']
        table['GDP per capita'] = table['GDP'] / table['Population']
        max_gdp = table['GDP per capita'].max()
        max_density = table['Density'].max()
        max_gdp_region = table[table['GDP per capita'] == max_gdp].index[0]
        max_density_region = table[table['Density'] == max_density].index[0]
        print(f'\tCluster {i}')
        print(f'\t\tMax GDP: {max_gdp}\tRegion: {max_gdp_region}')
        print(f'\t\tMax Density: {max_density}\tRegion: {max_density_region}')

    fig = px.scatter_3d(
        df, x='GDP per capita', y='Area', z='Population',
        color=kmeans.labels_, hover_data=['Country Name', 'Region', 'CO2 emission', 'Density'],
        width=1000, height=800,
        title=f'N-Clusters {n}'
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.show()

plt.show()
