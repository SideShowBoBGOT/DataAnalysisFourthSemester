import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import packaging

df = pd.read_csv(
    'data/Data4.csv', encoding='windows-1251', sep=';', decimal=','
).rename(columns={'Unnamed: 0': 'Country'})
features = df[['Ie', 'Iec', 'Is']]
kmeans = KMeans(
    init='random',
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)
print(kmeans.fit(features))
print(kmeans.cluster_centers_)
fig = px.scatter_3d(
    df, x='Ie', y='Iec', z='Is',
    color=kmeans.labels_, hover_data=['Country', 'Cql'],
    width=1000, height=800,
    title='Countries clusters by Ie, Iec, Is'
)

centers = kmeans.cluster_centers_
fig.add_scatter3d(
    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
    mode='markers', opacity=0.7, marker_symbol='x',
    marker = dict(
        color='green'
    ),
    name='Cluster centers',
    hovertemplate =
    'Ie: %{x}<br>' +
    'Iec: %{y}<br>' +
    'Is: %{z}<br>'
)

fig.update(layout_coloraxis_showscale=False)

fig.show()
cql = df['Cql'].to_numpy().reshape(-1, 1)

cql_kmeans = KMeans(
    init='random',
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

cql_kmeans.fit(cql)
print(cql_kmeans.cluster_centers_)
print(cql_kmeans.labels_)
_, countries_count = np.unique(kmeans.labels_, return_counts=True)
for i in range(len(countries_count)):
    print(f'Кількість країн в {i + 1}-му кластері по Ie, Iec, Is: {countries_count[i]}')
_, cql_countries_count = np.unique(cql_kmeans.labels_, return_counts=True)

for i in range(len(cql_countries_count)):
    print(f'Кількість країн в {i + 1}-му кластері по Cql: {cql_countries_count[i]}')


