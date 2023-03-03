import math

from my_hypothesis import *
from my_distributions_like import *


def calc_b_1(nums_one: list[float], nums_two: list[float]):
    n = len(nums_one)
    b_1 = n * sum([x * y for x, y in zip(nums_one, nums_two)])
    sum_one = sum(nums_one)
    sum_2 = sum([x ** 2 for x in nums_one])
    sum_squared = sum_one ** 2
    b_1 -= sum(nums_one) * sum(nums_two)
    b_1 /= (n * sum_2 - sum_squared)
    return b_1


def statistics_2d_to_nums(table: list[tuple[float, float, int]]) -> tuple[list[float], list[float]]:
    nums_one = []
    nums_two = []
    for el in table:
        one = el[0]
        two = el[1]
        n = el[2]
        for i in range(n):
            nums_one.append(one)
            nums_two.append(two)
    return nums_one, nums_two


def is_nums_dependent(r: float, n: int, confidence_level: float):
    t_expected = r * math.sqrt(n - 2) / math.sqrt(1 - math.pow(r, 2))
    t_cr = students_distribution(confidence_level, n - 2)
    return t_expected < t_cr


def correlation_analysis(nums_one: list[float], nums_two: list[float]):
    n = len(nums_one)
    x_one = mean(nums_one)
    x_two = mean(nums_two)
    x_one_diff = [x - x_one for x in nums_one]
    x_two_diff = [x - x_two for x in nums_two]
    x_one_diff_2 = [x ** 2 for x in x_one_diff]
    x_two_diff_2 = [x ** 2 for x in x_two_diff]
    diff_mul = [x * y for x, y in zip(x_one_diff, x_two_diff)]
    s_2_one = sum(x_one_diff_2) / (n - 1)
    s_2_two = sum(x_two_diff_2) / (n - 1)
    sigma_one = math.sqrt(s_2_one)
    sigma_two = math.sqrt(s_2_two)
    b_1_one = calc_b_1(nums_one, nums_two)
    b_0_one = x_two - b_1_one * x_one
    b_1_two = calc_b_1(nums_two, nums_one)
    b_0_two = x_one - b_1_two * x_two
    b_one_two = math.sqrt(b_1_one * b_1_two)
    cov = sum(diff_mul)
    r_xy = cov / ((n - 1) * sigma_one * sigma_two)
    k_xy = b_1_one * s_2_one
    print(f'n: {n}')
    print(f'x_one: {x_one}')
    print(f'x_two: {x_two}')
    print(f's_2_one: {s_2_one}')
    print(f's_2_two: {s_2_two}')
    print(f'sigma_one: {sigma_one}')
    print(f'sigma_two: {sigma_two}')
    print(f'correlation moment: cov: {cov}')
    print(f'sample correlation coefficient: r_xy: {r_xy}')
    print(f'k_xy: {k_xy}')
    print(f'b_1_one: {b_1_one}')
    print(f'b_0_one: {b_0_one}')
    print(f'b_1_two: {b_1_two}')
    print(f'b_0_two: {b_0_two}')
    print(f'b_one_two: {b_one_two}')
    print(f'y = ( {b_0_one} ) + ( {b_1_one} ) * x')
    print(f'x = ( {b_0_two} ) + ( {b_1_two} ) * y')
    print(f'is_dependent: {is_nums_dependent(r_xy, n, 0.001)}')


s_one = '1 2 3 4 6 7 8 10'
s_two = '15 16 18 19 21 22 25 27'
nums_one = [float(el) for el in s_one.split(' ')]
nums_two = [float(el) for el in s_two.split(' ')]

correlation_analysis(nums_one, nums_two)

# y = ( 13.557168784029038 ) + ( 1.3303085299455535 ) * x