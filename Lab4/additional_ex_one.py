import math
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np


from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

# cities
cities = pd.read_csv('data/worldcities.csv')
cities = cities[(cities['country'] == 'Ukraine') & ((cities['capital'] == 'primary') | (cities['capital'] == 'admin'))]
cities = gpd.GeoDataFrame(cities, geometry=gpd.points_from_xy(cities.lng, cities.lat))


def plot_cities(axis):
    cities.plot(ax=axis, color='black')
    for x, y, label in zip(cities.geometry.x, cities.geometry.y, cities.city):
        axis.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")


# population bubbles
cities_bubbles = cities.copy()
cities_bubbles['geometry'] = cities_bubbles['geometry'].centroid
country = gpd.read_file('data/ukraine/ukr_admbnda_adm1_sspe_20230201.shp')
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('Population')
country.plot(ax=axis)

cities_bubbles['population'] /= 1000
cities_bubbles.plot(ax=axis, column='population', markersize='population',
                    alpha=0.7, categorical=False, legend=True)
plot_cities(axis)


differences = []
for index_one, row_one in cities.iterrows():
    for index_two, row_two in cities.iterrows():
        sq_diff_lat = math.pow(row_one.lat - row_two.lat, 2)
        sq_diff_lng = math.pow(row_one.lng - row_two.lng, 2)
        dist_geo = math.sqrt(sq_diff_lat + sq_diff_lng)
        dist_km = haversine(row_one.lng, row_one.lat, row_two.lng, row_two.lat)
        differences.append((row_one.city, row_two.city, dist_geo, dist_km))

# max distance
differences.sort(key=lambda x: x[2])
max_diff = differences[-1]
print(max_diff)


# 2 GDP and Wage
gdp = pd.read_csv('data/ukr_GDP.csv')

def build_regression_empty_value():
    for index, row in gdp.iterrows():
        nums = row.iloc[1:]
        positive = [(int(x) - 2006, y) for x, y in nums.items() if y >= 0]
        y = [y for x, y in positive]
        y = np.array(y)
        x = [x for x, y in positive]
        pow_deg = 2
        pow_x = [pow_deg for _ in x]
        x = np.array(x)
        degree = 2
        model = np.poly1d(np.polyfit(np.power(x, np.array(pow_x)), y, degree))
        for column, element in nums.items():
            if element < 0:
                gdp.at[index, column] = model(int(column) - 2006)

build_regression_empty_value()
wages = pd.read_csv('data/ukr_ZP.csv')

country_gdp = country.copy()
country_wages = country.copy()

country_gdp = country_gdp.merge(gdp, how='left', on='ADM1_EN')
country_wages = country_wages.merge(wages, how='left', on='ADM1_EN')

    # GDP
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('2016 GDP per region')
country_gdp.plot(ax=axis, column='2016', categorical=False, legend=True)
plot_cities(axis)


    # Wage
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('2016 Wage per region')
country_wages.plot(ax=axis, column='2016', categorical=False, legend=True)
plot_cities(axis)


fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('Correlation between GDP and Wage')
correlation = country.copy()
gdp_raw = gdp.loc[:, '2006':'2016']
wages_raw = wages.loc[:, '2006':'2016']
correlation_raw = gdp.corrwith(wages_raw, axis=1, numeric_only=True)
correlation['Correlation'] = correlation_raw
correlation.plot(ax=axis, column='Correlation', categorical=False, legend=True)


plt.show()
