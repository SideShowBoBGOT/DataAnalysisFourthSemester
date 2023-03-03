import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import my_distributions_like as mydl
import my_statistic_values as myst
import numpy as np
import matplotlib
from my_colors import bcolors



def colorize(msg: str, color_type: str, is_flush_color: bool = True) -> str:
    colored_msg = f'{color_type}{msg}'
    if is_flush_color:
        colored_msg += f'{bcolors.ENDC}'
    return colored_msg


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


def print_normality_check_result(sig_lev: float, crit_val: float, res_stat: float):
    res_val_msg = colorize(f'{res_stat}', bcolors.OKCYAN)
    crit_val_msg = colorize(f'{crit_val}', bcolors.OKCYAN)
    sig_val_msg = colorize(f'{sig_lev}', bcolors.OKCYAN)
    prob_normal = colorize('\t\tProbably Normal:', bcolors.OKGREEN)
    prob_not_normal = colorize('\t\tProbably Not Normal:', bcolors.FAIL)
    result_msg = colorize(f'\t\t\tExpected Value', bcolors.WARNING)
    critical_msg = colorize('\t\t\tCritical Value', bcolors.WARNING)
    significance_level_msg = colorize('\t\t\tSignificance Level', bcolors.WARNING)
    if res_stat < crit_val:
        print(f'{prob_normal}')
    else:
        print(f'{prob_not_normal}')
    print(result_msg, res_val_msg)
    print(critical_msg, crit_val_msg)
    print(significance_level_msg, sig_val_msg)


def check_normally_distributed(dataset: pd.DataFrame, column_name: str) -> None:
    result = stats.anderson(dataset[column_name])
    for i in range(len(result.critical_values)):
        sig_lev, crit_val = result.significance_level[i], result.critical_values[i]
        print_normality_check_result(sig_lev, crit_val, result.statistic)


def dataframe_col_to_interval_discrete(dataset: pd.DataFrame, column_name: str, step: float = 0) -> list[float]:
    nums_count_freq = myst.nums_count_frequency_tuples(dataset[column_name].values.tolist())
    intervals = myst.interval_sequence(nums_count_freq, step=step)
    discrete = myst.discrete_sequence_from_interval_count_frequency(intervals)
    hist_data = [row[0] for row in discrete for _ in range(row[1])]
    return hist_data


def check_columns_normally_distributed(dataset: pd.DataFrame, columns: list[str]):
    print(colorize(f'Check for normal distribution:', bcolors.HEADER))
    fig, axes = plt.subplots(len(columns), 1)

    for i, col in enumerate(columns):
        print(colorize(f'\t{col}:', bcolors.OKBLUE))
        check_normally_distributed(dataset, col)
        hist_data = dataframe_col_to_interval_discrete(dataset, col)
        axes[i].hist(hist_data)
        axes[i].set_title(col)


def check_mean_median_columns(dataset: pd.DataFrame, columns: list[str]):
    print(colorize(f'Mean-median:', bcolors.HEADER))
    hypothesis_accepted = colorize(f'\t\tAccepted:', bcolors.OKGREEN)
    hypothesis_rejected = colorize(f'\t\tRejected:', bcolors.FAIL)
    for col in columns:
        mean = dataset[col].mean()
        median = dataset[col].median()
        t_statistic, p_value = stats.ttest_1samp(a=dataset[col], popmean=median)
        mean_msg = colorize('\t\t\tMean: ', bcolors.WARNING)
        median_msg = colorize('\t\t\tMedian: ', bcolors.WARNING)
        mean_value = colorize(f'{mean}', bcolors.OKCYAN)
        median_value = colorize(f'{median}', bcolors.OKCYAN)
        colored_col = colorize(f'{col}:', bcolors.OKBLUE)
        print(f'\t{colored_col}')
        if p_value < 0.01:
            print(hypothesis_rejected)
        else:
            print(hypothesis_accepted)
        print(f'{mean_msg}{mean_value}')
        print(f'{median_msg}{median_value}')


def group_by_column_normally_distibuted(dataset: pd.DataFrame, group_column: str, normally_checked: str):
    print(colorize(f'Normally checked {normally_checked} by {group_column}:', bcolors.HEADER))
    diffs = []
    groups = list(set(dataset[group_column].values.tolist()))
    for val in groups:
        data = dataset[dataset[group_column] == val]
        result = stats.anderson(data[normally_checked])
        expected = result.statistic
        sig_lev, crit_val = result.significance_level[0], result.critical_values[0]
        print(colorize(f'\t{val}:', bcolors.OKBLUE))
        print_normality_check_result(sig_lev, crit_val, expected)
        difference_msg = colorize('\t\t\tDifference: ', bcolors.WARNING)
        difference = abs(expected - crit_val)
        diff_val_msg = colorize(f'{difference}', bcolors.OKCYAN)
        diffs.append(difference)
        print(difference_msg, diff_val_msg)
    min_diff = min(diffs)
    index = diffs.index(min_diff)
    min_diff_msg = colorize('\tMin Difference: ', bcolors.OKBLUE)
    min_diff_key_val = groups[index]
    min_diff_key = colorize(f'{min_diff_key_val}', bcolors.WARNING)
    print(min_diff_msg, min_diff_key)



if __name__ == '__main__':
    data_path = 'data/Data2.csv'
    dataset = read_dataset(data_path, sep=';')
    numeric_cols = dataset.columns[2:]

    for column_name in numeric_cols:
        replace_comma_with_dots(dataset, column_name)
        convert_column_to_float(dataset, column_name)
        replace_nan_with_mean(dataset, column_name)
        convert_float_with_positive(dataset, column_name)

    dataset['GDP'] = dataset['GDP per capita'] * dataset['Populatiion']
    dataset['Population'] = dataset['Populatiion']
    dataset.drop(['Populatiion'], axis=1)
    numeric_cols = dataset.columns[2:]

    # ['GDP per capita', 'Populatiion', 'CO2 emission', 'Area', 'GDP', 'Population']
    # 2 Вказати, чи є параметри, що розподілені за нормальним законом
    check_columns_normally_distributed(dataset, numeric_cols)

    # 3 Перевірити гіпотезу про рівність середнього і медіани для одного з параметрів
    check_mean_median_columns(dataset, numeric_cols)

    # 4 Вказати, в якому регіоні розподіл викидів СО2 найбільш близький до нормального
    group_by_column_normally_distibuted(dataset, 'Region', 'CO2 emission')

    df_group_by_region = dataset.groupby('Region')
    df_region_population = df_group_by_region.sum(numeric_only=True)[['Population']]
    fig, ax = plt.subplots(1, 1)
    ax.pie(df_region_population['Population'], labels=df_region_population.index)

    plt.show()
