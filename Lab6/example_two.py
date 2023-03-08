import pandas as pd
import numpy as np

# Візуалізація
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Табличний вивід
from tabulate import tabulate

# Визначення точки локтя (кластеризація)
from kneed import KneeLocator

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import scipy.stats as stats

df = pd.read_csv('data/pca.csv', delimiter=';', decimal=',', encoding='cp1252')
df.info()
df.head(10)
df.isna().any()
indexes_abbr_names = df[['index_abbr', 'index_name', 'better_value']].drop_duplicates().reset_index(drop=True)
print('dddd', df.columns)
dropped_df = df.drop(columns=['abbr_reg', 'index_name'])
print('dddd', dropped_df.columns)

dropped_df.drop_duplicates(inplace=True)
pivoted_df = dropped_df.pivot_table(index=['year', 'name_reg'], columns=['index_abbr'], values='val')
print(pivoted_df)
pivoted_df.columns.rename('id', inplace=True)
pivoted_df.reset_index(inplace=True)

def get_missing_data_stats(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

missing_data = get_missing_data_stats(pivoted_df)
missing_data.head(5)

cols_to_drop = missing_data[missing_data['Percent'] > 30].index
pivoted_df.drop(columns=cols_to_drop, inplace=True)
imputer = KNNImputer(n_neighbors=20)
pivoted_df[pivoted_df.columns[2:]] = imputer.fit_transform(pivoted_df[pivoted_df.columns[2:]])

get_missing_data_stats(pivoted_df).sum()

scaler = StandardScaler()
pivoted_df[pivoted_df.columns[2:]] = scaler.fit_transform(pivoted_df[pivoted_df.columns[2:]])

corrmat = pivoted_df.corr()

f, ax = plt.subplots(figsize=(20, 18))

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()



