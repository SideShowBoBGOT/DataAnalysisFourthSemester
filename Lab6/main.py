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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


#correct file
# with open('data/titanic.csv') as input_file:
#     with open('data/data_titanic.csv', 'w') as output_file:
#         lines = []
#         statuses = []
#         for index, line in enumerate(input_file):
#             if index == 0:
#                 lines.append(line.replace(',', ';'))
#                 continue
#             parts = line.split('"')
#             parts[0] = parts[0].replace(',', ';')
#             parts[2] = parts[2].replace(',', ';')
#             statuses.append(parts[1].split(',')[1].split('.')[0].lstrip())
#             lines.append(''.join(parts))
#         statuses = list(set(statuses))
#         output_file.writelines(lines)

# print(statuses)

status_gender_pairs = {
    'Rev': 'male',
    'Miss': 'female',
    'Ms': 'male',
    'Dr': 'male',
    'Mlle': 'female',
    'Col': 'male',
    'the Countess': 'female',
    'Jonkheer': 'male',
    'Don': 'male',
    'Capt': 'male',
    'Master': 'male',
    'Mme': 'female',
    'Mrs': 'female',
    'Mr': 'male',
    'Lady': 'female',
    'Sir': 'male',
    'Major': 'male'
}

df = pd.read_csv('data/data_titanic.csv', delimiter=';', decimal='.')
# drop unnecessary columns

df = df.drop(['PassengerId'], axis=1)

# Take a look on what data is missing
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

percent_1 = df.isnull().sum() / df.isnull().count() * 100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([percent_2], axis=1, keys=['%'])
plot_rows = 1
plot_cols = 1
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(8, 6))
msno.matrix(df, ax=axis)

# empty values by percentage in column
fig, axis = plt.subplots(plot_rows, plot_cols, figsize=(8, 6))
axis.bar(missing_data.index, missing_data.iloc[:, 0])

df = df.drop(['Cabin'], axis=1)

# fill Nan "Age"-values with mean
df_sex_mean_age = pd.pivot_table(df, index='Sex', values='Age', aggfunc='mean')
print(df_sex_mean_age)
mean_age_female, mean_age_male = df_sex_mean_age.iloc[0, 0], df_sex_mean_age.iloc[1, 0]


# fill gender
df['Sex'] = df.apply(lambda x: status_gender_pairs[x['Name'].split(',')[1].split('.')[0].lstrip()], axis=1)
df['Gender'] = df.apply(lambda x: x['Sex'] == 'male', axis=1)

def mean_age_functor(row):
    if str(row['Age']) != 'nan':
        return row['Age']
    if row['Sex'] == 'male':
        return mean_age_male
    return mean_age_female


df['Age'] = df.apply(mean_age_functor, axis=1)

# fill "Embarked" with the most frequent value
fig, axis = plt.subplots(figsize=(8, 6))
embarked_counts = df['Embarked'].value_counts()
axis.pie(embarked_counts, labels=embarked_counts.index, autopct='%1.1f%%')

# fill empty embarked with "S"
df['Embarked'].fillna('S', inplace=True)
df['EmbarkedValue'] = df.apply(lambda x: ['S', 'Q', 'C'].index(x['Embarked']), axis=1)

# fill "Fare" with mean
df['Fare'].fillna(value=df['Fare'].mean(), inplace=True)
df['SibSp'].fillna(value=0, inplace=True)
df['Parch'].fillna(value=0, inplace=True)
df['Relatives'] = df['SibSp'] + df['Parch']
df = df.drop(['Parch', 'SibSp'], axis=1)

# Since the Ticket attribute has 681 unique tickets,
# it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.
print(df['Ticket'].describe())
df = df.drop(['Ticket'], axis=1)

# plot correlation matrix
fig, axis = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), ax=axis, annot=True)


# plot male-female Age statistic
fig = sns.FacetGrid(df, hue='Gender', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# plot male-female Count statistic
fig, axis = plt.subplots(figsize=(8, 6))
male_female_counts = df['Sex'].value_counts()
axis.pie(male_female_counts, labels=male_female_counts.index, autopct='%1.1f%%')


factors = df.columns.values.tolist()
factors = [f for f in factors if f not in ['Sex', 'Gender', 'Survived', 'Name', 'Embarked']]

# plot male-female Survival Rate based on Age statistic

fig, axis = plt.subplots(nrows=len(factors), ncols=2, figsize=(10, 8))
for row, factor in enumerate(factors):
    for col, el in enumerate(['male', 'female']):
        ax = axis[row][col]
        gender = df[df['Sex'] == el]
        survived_age = gender[gender['Survived'] == 1][factor]
        dead_age = gender[gender['Survived'] == 0][factor]
        sns.distplot(survived_age, color='green', bins=18, label='survived', ax=ax, kde=False)
        sns.distplot(dead_age, color='red', bins=40, label='dead', ax=ax, kde=False)
        ax.set_title(factor)
        ax.set_xlabel(el)


plt.show()
