import math
from sympy.functions.special.error_functions import erf, erfinv
import scipy.stats as stats


def laplace_func(arg: float) -> float:
    return erf(arg / 2 ** 0.5) / 2


def inverse_laplace(value: float) -> float:
    return erfinv(value * 2) * 2 ** 0.5


# a.k.a in official math it is called: "Finding the T-value of a P-Value"
# https://www.w3schools.com/statistics/statistics_students_t_distribution.php
# here it is needed to convert from two-tail to one-tail table
def students_distribution(value: float, freedom_degrees: int) -> float:
    return stats.t.ppf(q=1 - (1 - value) / 2, df=freedom_degrees)


# a.k.a in official math it is called: "Finding the P-Value of a T-Value"
# https://www.w3schools.com/statistics/statistics_students_t_distribution.php
# here it is needed to convert from two-tail to one-tail table
def inverse_students_distribution(value: float, freedom_degrees: int) -> float:
    return 1 - 2 * (1 - stats.t.cdf(value, freedom_degrees))


def f_distribution(degrees_freedom_one: int, degrees_freedom_two: int, confidence_level: float) -> float:
    return stats.f.ppf(1 - confidence_level, degrees_freedom_one, degrees_freedom_two)


def inverse_f_distribution(n_one: int, n_two: int, value: float) -> float:
    return stats.f.cdf(value, n_one, n_two)


def chi_squared_distribution(confidence_level: float, freedom_degrees: int) -> float:
    return stats.chi2.ppf(confidence_level, freedom_degrees)


def inverse_chi_squared_distribution(value: float, freedom_degrees: int) -> float:
    return stats.chi2.cdf(value, freedom_degrees)


def x_b_trust_interval(xb, probability, s2, n) -> tuple[float, float]:
    tt = inverse_laplace(probability / 2) * math.sqrt(s2 / n)
    print(tt)
    left = xb - tt
    right = xb + tt
    return left, right


def dispersion_trust_interval(n: int, x_squared_1: float, x_squared_2: float, s2) -> tuple[float, float]:
    top = (n - 1) * s2
    left = top / x_squared_1
    right = top / x_squared_2
    return left, right


def mean_square_dispersion_trust_interval(n, probability, s2) -> tuple[float, float]:
    left, right = dispersion_trust_interval(n, probability, s2)
    return left ** 0.5, right ** 0.5


def prob_pairs_for_x_squared(probability: float):
    return (1 - probability) / 2, (1 + probability) / 2
