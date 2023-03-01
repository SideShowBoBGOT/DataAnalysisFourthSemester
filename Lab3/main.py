import pandas as pd
import matplotlib.pyplot as plt

from my_colors import bcolors


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


if __name__ == "__main__":
    data_path = 'data/Data2.csv'
    dataset = read_dataset(data_path, sep=';')
    # https://www.geeksforgeeks.org/how-to-fill-nan-values-with-mean-in-pandas/
    # ['Country', 'ISO', 'Conflicts intencity', 'Hospital beds',
    # 'High-technology exports', 'GDP per capita', 'Population']
    # for column_name in dataset.columns[2:]:
    #     dataset = replace_comma_with_dots(dataset, column_name)

    # 1 Чи є пропущені значення? Якщо є, замінити середніми
    for column_name in dataset.columns[2:]:
        replace_comma_with_dots(dataset, column_name)
        convert_column_to_float(dataset, column_name)
        replace_nan_with_mean(dataset, column_name)
        convert_float_with_positive(dataset, column_name)

    dataset['GDP'] = dataset['GDP per capita'] * dataset['Populatiion']

    # 2 Яка країна має найбільший ВВП на людину (GDP per capita)? Яка має найменшу площу?
    df_max_gdp = dataset.nlargest(1, ['GDP per capita'])
    max_gdp = df_max_gdp['GDP per capita'].values[0]
    country_max_gdp = df_max_gdp['Country Name'].values[0]
    print(f'{bcolors.HEADER}Max GDP:\n{bcolors.ENDC}'
          f'\t{bcolors.OKBLUE}Country: '
          f'{bcolors.OKGREEN}{country_max_gdp}\t'
          f'\t{bcolors.OKBLUE}GDP: {bcolors.OKGREEN}{max_gdp}')

    df_min_area = dataset.nsmallest(1, ['Area'])
    min_area = df_min_area['Area'].values[0]
    country_min_area = df_min_area['Country Name'].values[0]
    print(f'{bcolors.HEADER}Min Area:\n{bcolors.ENDC}'
          f'\t{bcolors.OKBLUE}Country: '
          f'{bcolors.OKGREEN}{country_min_area}\t'
          f'\t{bcolors.OKBLUE}Area: '
          f'{bcolors.OKGREEN}{min_area}')

    # 3 В якому регіоні середня площа країни найбільша?
    df_group_by_region = dataset.groupby('Region')
    df_region_area_sum = df_group_by_region.sum(numeric_only=True)[['Area']]
    df_region_count = df_group_by_region.count()[['Area']]
    df_average_area = df_region_area_sum / df_region_count
    df_max_average_area = df_average_area.nlargest(1, ['Area'])
    max_average_area_region = df_max_average_area.axes[0][0]
    max_average_area = df_max_average_area['Area'].values[0]
    print(f'{bcolors.HEADER}Max average area:{bcolors.ENDC}\n'
          f'\t{bcolors.OKBLUE}Region: '
          f'{bcolors.OKGREEN}{max_average_area_region}\t'
          f'\t{bcolors.OKBLUE}Value: '
          f'{bcolors.OKGREEN}{max_average_area}')

    # 5 Чи співпадає в якомусь регіоні середнє та медіана ВВП?
    df_region_gdp_mean = pd.DataFrame()
    df_region_gdp_average = pd.DataFrame()
    df_region_gdp_mean['Mean'] = df_group_by_region.mean(numeric_only=True)[['GDP per capita']]
    df_region_gdp_average['Average'] = df_group_by_region.median(numeric_only=True)[['GDP per capita']]
    df_region_mean_average = pd.concat([df_region_gdp_mean, df_region_gdp_average], axis=1)
    df_region_mean_average['Difference'] = df_region_mean_average['Mean'] - df_region_mean_average['Average']
    df_region_mean_average['Difference'] = df_region_mean_average['Difference'].abs()
    df_smallest_mean_average_difference = df_region_mean_average.nsmallest(1, ['Difference'])
    mean_average_info = bcolors.HEADER + 'Mean-Average Equality:\n' + bcolors.ENDC
    mean_average_info += f'\t{bcolors.OKBLUE}Region: {bcolors.OKGREEN}{df_smallest_mean_average_difference.axes[0][0]}'
    for column_name in ['Mean', 'Average', 'Difference']:
        mean_average_info += f'\t\t{bcolors.OKBLUE}{column_name}: ' \
                             f'{bcolors.OKGREEN}' \
                             f'{df_smallest_mean_average_difference[column_name].values[0]}' \
                             f'{bcolors.ENDC}'
    print(mean_average_info)

    # 6 Вивести топ 5 країн та 5 останніх країн по ВВП та кількості СО2 на душу населення.
    dataset_gdp_desc = dataset.sort_values('GDP per capita', ascending=False)
    dataset_gdp_asc = dataset.sort_values('GDP per capita', ascending=True)
    dataset_co2_desc = dataset.sort_values('CO2 emission', ascending=False)
    dataset_co2_asc = dataset.sort_values('CO2 emission', ascending=True)
    print(f'{bcolors.HEADER}GDP Top 5:{bcolors.ENDC}\n', dataset_gdp_desc.head(5).to_string())
    print(f'{bcolors.HEADER}GDP Bottom 5:{bcolors.ENDC}\n', dataset_gdp_asc.head(5).to_string())
    print(f'{bcolors.HEADER}CO2 Top 5:{bcolors.ENDC}\n', dataset_co2_desc.head(5).to_string())
    print(f'{bcolors.HEADER}CO2 Bottom 5:{bcolors.ENDC}\n', dataset_co2_asc.head(5).to_string())



