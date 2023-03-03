from my_statistic_values import *
from my_distributions import *


# 6.1
def my_round(number: float) -> int:
    number_int = int(number)
    diff = number - number_int
    if diff < 0.5:
        return number_int
    return number_int + 1


def even_distribution_function_generator(mean: float, deviation: float):
    values = [mean - math.sqrt(3 * deviation), mean + math.sqrt(3 * deviation)]
    a = min(values)
    b = max(values)
    return lambda x: (x - a) / (b - a), 2


def exponential_distribution_function_generator(mean: float, deviation: float):
    lam = 1 / mean
    return lambda x: lam * math.pow(math.e, lam * x), 1


def normal_distribution_function_not_normalized_values_generator(mean: float, deviation: float):
    return lambda x: laplace_func((x - mean) / deviation), 2


def binomial_distribution_function_generator(mean: float, deviation: float):
    p = 1 - deviation / mean
    q = 1 - p
    n = int(math.pow(mean, 2) / (mean - deviation))
    return lambda x: math.comb(n, x) * math.pow(p, x) * math.pow(q, n - x), 2


def geometric_distribution_function_generator(mean: float, deviation: float):
    p = 1 / mean
    return lambda x: p * math.pow(1 - p, x - 1)


def poisson_distribution_function_generator(mean: float, deviation: float):
    lam = mean
    e_lam = math.pow(math.e, -lam)
    return lambda x: math.pow(lam, x) * e_lam / math.factorial(x)


def get_distribution_function(discrete: list[tuple[float, int]], distribution_function_generator):
    n = sum(x[1] for x in discrete)
    mean = x_b_from_nums_count(discrete)  # x_star
    x_star_squared = sum(math.pow(x[0], 2) * x[1] for x in discrete) / n
    deviation = math.sqrt(x_star_squared - math.pow(mean, 2))  # sigma_star
    return distribution_function_generator(mean, deviation)


def get_sequence_distributed_like_result(
        probabilities: list[float], frequencies: list[float],
        n: int, l: int, number_of_parameters: int, confidence_level: float) -> bool:
    frequencies_pairs: list[tuple[float, float]] = []
    for probability, freq in zip(probabilities, frequencies):
        theoretical_freq = probability * n
        frequencies_pairs.append((freq, theoretical_freq))
    freq_diffs_squared = list(
        map(lambda pair: math.pow(pair[0] - pair[1], 2), frequencies_pairs)
    )
    x_2_expected = 0
    for diff_2, pair in zip(freq_diffs_squared, frequencies_pairs):
        x_2_expected += diff_2 / pair[1]
    degrees_freedom = l - (number_of_parameters + 1)
    x_2_critical = chi_squared_distribution(1 - confidence_level, degrees_freedom)
    result = x_2_expected < x_2_critical
    return result


# https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
def is_intervals_sequence_distributed_like(
        intervals: list[tuple[float, float, int]],
        distribution_function_generator,
        confidence_level: float
) -> bool:
    discrete = discrete_sequence_from_interval_count(intervals)
    n = sum(x[1] for x in discrete)
    func, number_of_parameters = get_distribution_function(discrete, distribution_function_generator)
    l = len(intervals)
    probabilities: list[float] = []
    frequencies: list[float] = []
    is_normal = distribution_function_generator == normal_distribution_function_not_normalized_values_generator
    for i, el in enumerate(intervals):
        one = el[0]
        two = el[1]
        if is_normal and i == 0:
            one = -math.inf
        elif is_normal and i == len(intervals) - 1:
            two = math.inf
        f_one = func(one)
        f_two = func(two)
        probability = f_two - f_one
        probabilities.append(probability)
        frequencies.append(el[2])
    result = get_sequence_distributed_like_result(
        probabilities, frequencies, n, l, number_of_parameters, confidence_level)
    return result


# https://real-statistics.com/tests-normality-and-symmetry/statistical-tests-normality-symmetry/kolmogorov-smirnov-test/
def is_intervals_sequence_normally_distributed(intervals: list[tuple[float, float, int]],
                                               confidence_level: float) -> bool:
    """
        Example from the book( page 45 ):
            confidence_level = 0.05
            intervals_count = [
                (3, 8, 6),
                (8, 13, 8),
                (13, 18, 15),
                (18, 23, 40),
                (23, 28, 16),
                (28, 33, 8),
                (33, 38, 7)
            ]
            x_star = 20.7
            sigma = 7.28
            x_2_expected = 12.2281
            x_2_critical = 9.5
            result = x_2_expected < x_2_critical // False
    """
    result = is_intervals_sequence_distributed_like(
        intervals,
        normal_distribution_function_not_normalized_values_generator,
        confidence_level
    )
    return result


# https://real-statistics.com/other-key-distributions/exponential-distribution/
def is_intervals_sequence_exponentially_distributed(intervals: list[tuple[float, float, int]],
                                                    confidence_level: float) -> bool:
    result = is_intervals_sequence_distributed_like(
        intervals,
        exponential_distribution_function_generator,
        confidence_level
    )
    return result


# 6.2
def kolmogorov_func(lamda: float) -> float:
    total = 0
    lamda_2 = lamda ** 2
    for k in range(-10000, 10000):
        k_2 = k ** 2
        total += math.pow(-1, k) * math.pow(math.e, -2 * lamda_2 * k_2)
    result = 1 - total
    return result


def kolmogorov_smirnov(intervals: list[tuple[float, float, int]], arg_function) -> float:
    """
        intervals = [
            (0, 5, 30),
            (5, 10, 16),
            (10, 15, 7),
            (15, 20, 3),
            (20, 25, 2),
            (25, 30, 1),
            (30, 35, 1)
        ]
        arg_func = lambda x: 1 - math.pow(math.e, -0.14 * x)

        # in the book values were too much rounded
        lamda = 0.102739

        result = 1.0
    """

    n = sum(x[2] for x in intervals)
    total = 0
    f_value_pairs: list[tuple[float, float]] = []
    for row in intervals:
        total += row[2]
        f_value_pairs.append((total / n, arg_function(row[1])))
    d_n = max(list(map(lambda x: math.fabs(x[0] - x[1]), f_value_pairs)))
    value = d_n * math.sqrt(n)
    result = kolmogorov_func(value)
    return result


s = '75 85 84 81 84 80 82 76 75 77 80 82 81 84 85 ' \
    '77 76 84 83 84 78 77 88 86 87 79 80 79 78 87 ' \
    '78 77 88 86 84 79 80 79 78 81'

nums = float_sequence_from_str(s, ' ')
nums_count = nums_count_frequency_tuples(nums)
intervals = interval_sequence(nums_count)
print(is_intervals_sequence_normally_distributed(
    intervals,
    0.05
))