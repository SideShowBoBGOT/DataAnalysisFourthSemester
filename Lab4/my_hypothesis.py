from my_statistic_values import *
from my_distributions import *

# 5.2 one_sample_t_student_criterion
def is_mathematical_expectation_equals_value(nums: list[float], xb_expected: float, confidence_level: float) -> bool:
    """
    Checks whether M(X) a.k.a mathematical expectation equals expected value
    return:
        case true a.k.a t_p < t_alpha:
            mathematical expectation is equal expected value
        case false a.k.a t_p >= t_alpha:
            mathematical expectation is NOT equal to expected value
    """

    xb = mean(nums)
    d = variance(nums)
    n = len(nums)
    std_error = math.sqrt(d / n)
    t_p = (xb - xb_expected) / std_error
    degrees_freedom = n - 1
    t_alpha = students_distribution(1 - confidence_level, degrees_freedom)
    result = t_p < t_alpha
    return result


# 5.3 two_sample_t_student_criterion_independent
def is_mathematical_expectations_equal_independent(nums_one: list[float], nums_two: list[float],
                                                   confidence_level: float) -> bool:
    """
    Checks whether mathematical expectation of both sequences are equal.
    !!!Sequences must be independent!!!

    return:
        case true a.k.a t_p < t_alpha:
            mathematical expectation of both sequences are equal
        case false a.k.a t_p >= t_alpha:
            mathematical expectation of both sequences are NOT equal
    """

    n_one = len(nums_one)
    n_two = len(nums_two)
    xb_one = mean(nums_one)
    xb_two = mean(nums_two)
    d_one = variance(nums_one)
    d_two = variance(nums_two)
    std_error = math.sqrt(d_one / n_one + d_two / n_two)
    t_p = math.fabs(xb_one - xb_two) / std_error
    degrees_freedom = (n_one - 1) + (n_two - 1)
    t_alpha = students_distribution(1 - confidence_level, degrees_freedom)
    result = t_p < t_alpha
    return result


# 5.4 two_sample_t_student_criterion_dependent
def is_mathematical_expectations_equal_dependent(nums_one: list[float], nums_two: list[float],
                                                 confidence_level: float) -> bool:
    """
    Checks whether mathematical expectation of both sequences are equal.
    !!!Sequences must be dependent!!!

    return:
        case true a.k.a t_p < t_alpha:
            mathematical expectation of both sequences are equal
        case false a.k.a t_p >= t_alpha:
            mathematical expectation of both sequences are NOT equal
    """
    diff_tuples = list(zip(nums_one, nums_two))
    diff_sequence = [one - two for one, two in diff_tuples]
    xb = mean(diff_sequence)
    # I don`t know why in 5.4( page 35 ) it is divided by n - 1
    # instead of n for dispersion, so I use s_squared
    # so take this matter to consideration
    d = sample_variance(diff_sequence)
    n = len(diff_sequence)
    std_error = math.sqrt(d / n)
    degrees_freedom = n - 1
    t_p = math.fabs(xb) / std_error
    t_alpha = students_distribution(1 - confidence_level, degrees_freedom)
    result = t_p < t_alpha
    return result


# 5.5 f_criterion
def is_dispersions_equal(nums_one: list[float], nums_two: list[float],
                         confidence_level: float) -> bool:
    """
    Checks whether dispersions of both sequences are equal.

    return:
        case true a.k.a f_expected < f_critical:
            dispersions of both sequences are equal
        case false a.k.a f_expected >= f_critical:
            dispersions of both sequences are NOT equal
    """

    s_2_one = sample_variance(nums_one)
    s_2_two = sample_variance(nums_two)
    n_one = len(nums_one)
    n_two = len(nums_two)
    degrees_freedom_one = n_one - 1
    degrees_freedom_two = n_two - 1
    mmin = min(s_2_one, s_2_two)
    mmax = max(s_2_one, s_2_two)
    f_expected = mmax / mmin
    f_critical = f_distribution(degrees_freedom_one, degrees_freedom_two, confidence_level)
    result = f_expected < f_critical
    return result


# 5.6 check of hypothesis of sample correlation coefficient
def is_sequences_independent(nums_one: list[float], nums_two: list[float],
                             confidence_level: float) -> bool:
    """
    Checks whether sequences are independent. In other words,
    checks whether hypothesis of correlation being 0 is close
    to truth

    return:
        case true a.k.a t < t_critical:
            hypothesis of correlation being 0 is close to truth
        case false a.k.a t < t_critical:
            hypothesis of correlation being 0 is NOT close to truth
    """

    n_one = len(nums_one)
    n_two = len(nums_two)
    n = min(n_one, n_two)
    x = nums_one[:n]
    y = nums_two[:n]
    rb = correlation(x, y)
    degrees_freedom = n - 2
    t = math.fabs(rb * math.sqrt(n - 2) / math.sqrt(1 - math.pow(rb, 2)))
    t_critical = students_distribution(1 - confidence_level, degrees_freedom)
    result = t < t_critical
    return result


# 5.7 x_2_comparison_hypothetical_dispersion_with_general
# case one:
#   H_0 = s_squared == s_squared_expected
#   H_1 = s_squared > s_squared_expected
# yeah name of the function is stupid, but what can I do
def is_sample_dispersion_equal_to_general_reverse_greater(
        nums: list[float], expected_dispersion: float, confidence_level: float) -> float:
    n = len(nums)
    s2 = sample_variance(nums)
    x2_expected = (n - 1) * s2 / expected_dispersion
    degrees_freedom = n - 1
    x2_critical = chi_squared_distribution(confidence_level, degrees_freedom)
    result = x2_expected < x2_critical
    return result


def is_sample_dispersion_equal_to_general_reverse_not_equal(
        nums: list[float], expected_dispersion: float, confidence_level: float) -> float:
    n = len(nums)
    s2 = sample_variance(nums)
    x2_expected = (n - 1) * s2 / expected_dispersion
    degrees_freedom = n - 1
    x2_critical_right = chi_squared_distribution(1 - confidence_level / 2, degrees_freedom)
    x2_critical_left = chi_squared_distribution(confidence_level / 2, degrees_freedom)
    result = x2_critical_left < x2_expected < x2_critical_right
    return result


def is_sample_dispersion_equal_to_general_reverse_smaller(
        nums: list[float], expected_dispersion: float, confidence_level: float) -> float:
    n = len(nums)
    s2 = sample_variance(nums)
    x2_expected = (n - 1) * s2 / expected_dispersion
    degrees_freedom = n - 1
    x2_critical = chi_squared_distribution(1 - confidence_level, degrees_freedom)
    result = x2_expected > x2_critical
    return result


# s_one = '55, 55, 60, 46, 54, 57, 59, 60, 57, 46, '\
#         '55, 46, 60, 46, 54, 57, 60, 60, 57, 59, '\
#         '47, 47, 55, 46, 54, 60, 55, 54, 57, 59, ' \
#         '47, 57, 59, 46, 55, 54, 59, 59, 54, 60'
#
# s_two = '6,80 6,54 6,90 7,00 6,53 6,93 6,79 6,53 6,57 6,93 '\
#         '6,80 6,90 6,90 7,00 6,54 6,93 6,79 6,53 6,57 7,00 '\
#         '6,71 6,90 6,90 7,00 6,53 6,93 6,79 6,53 6,72 6,54 '\
#         '6,80 6,54 6,90 6,93 6,93 6,93 6,79 6,53 6,57 6,54'
#
# confidence_level = 0.05
# s_one = s_one.replace(' ', '')
# nums_one = float_sequence_from_str(s_one, delim=',')
# s_two = s_two.replace(',', '.')
# nums_two = float_sequence_from_str(s_two, ' ')
# print(nums_one)
# print(nums_two)
#
# print(is_mathematical_expectations_equal_independent(nums_one, nums_two, 0.05))
# print()