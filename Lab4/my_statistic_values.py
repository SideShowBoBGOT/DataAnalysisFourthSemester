import math

def mean(l: list[float]) -> float:
    total = 0
    for i in l:
        total += i
    total /= len(l)
    return total


def variance(l: list[float]) -> float:
    xb = mean(l)
    l_set = set(l)
    pp = list(map(lambda x: (x, l.count(x)), l_set))
    db = 0
    for el in pp:
        db += math.pow((el[0] - xb), 2) * el[1]
    db /= len(l)
    return db


def sample_variance(l: list[float]) -> float:
    db = variance(l)
    n = len(l)
    return db * n / (n - 1)


def sample_deviation(l: list[float]) -> float:
    return sample_variance(l) ** 0.5


# https://en.wikipedia.org/wiki/Correlation
def correlation(nums_one: list[float], nums_two: list[float]) -> float:
    if len(nums_one) != len(nums_two):
        raise ValueError
    xb_one = mean(nums_one)
    xb_two = mean(nums_two)
    sigma_one = sample_deviation(nums_one)
    sigma_two = sample_deviation(nums_two)
    n = len(nums_one)
    x_one_two = 0
    for element_one, element_two in zip(nums_one, nums_two):
        diff_one = (element_one - xb_one)
        diff_two = (element_two - xb_two)
        x_one_two += diff_one * diff_two
    r = x_one_two / ((n - 1) * (sigma_one * sigma_two))
    return r


def criterion_correlation(nums_one: list[float], nums_two: list[float]) -> float:
    r = correlation(nums_one, nums_two)
    n = len(nums_one)
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    return t


def float_sequence_from_str(s: str, delim=' ', is_coma_in_floats=False) -> list[float]:
    nums = s.split(delim)
    if is_coma_in_floats:
        nums = list(map(lambda x: x.replace(',', '.'), nums))
    nums = list(map(lambda el: float(el.lstrip().rstrip()), nums))
    nums.sort()
    return nums


def nums_count_frequency_tuples(nums: list[float]):
    nums_set = list(set(nums))
    nums_set.sort()
    return list(map(lambda x: (x, nums.count(x), nums.count(x) / len(nums)), nums_set))


def emperical_function(nums_count_frequency: list[tuple[float, int, float]]):
    print('EMPERICAL FUNCTION')
    total = 0
    size = len(nums_count_frequency)
    print(f'\t{0}, x <= {nums_count_frequency[0][0]}')
    for i in range(size - 1):
        total += nums_count_frequency[i][2]
        left = nums_count_frequency[i][0]
        right = nums_count_frequency[i + 1][0]
        print(f'\t{total}, {left} < x <= {right}')
    print(f'\t1, x > {nums_count_frequency[size - 1][0]}')


def cumulate_ogiva(nums_count_frequency: list[tuple[float, int, float]]):
    print('CUMULATE AND OGIVA')
    total = 0
    for i in nums_count_frequency:
        total += i[1]
        print(f'\t{i[0]}\t{i[1]}\t{total}')


def print_interval_sequence(intervals: list[tuple[float, float, int, float]]) -> None:
    print('INTERVALS')
    print(f'\tStep = {intervals[1][0] - intervals[0][0]}')
    print(f'\tMin = {intervals[0][0]}')
    print(f'\tMax = {intervals[len(intervals) - 1][0]}')
    for i in intervals:
        print(f'\t[{i[0]}; {i[1]})\t{i[2]}\t{i[3]}')


def interval_sequence(nums_count_frequency: list[tuple[float, int, float]], step=0) \
        -> list[tuple[float, float, int, float]]:
    size = len(nums_count_frequency)
    total_objects = sum(x[1] for x in nums_count_frequency)
    mmin = nums_count_frequency[0][0]
    mmax = nums_count_frequency[size - 1][0]
    if step == 0:
        step = (mmax - mmin) / math.floor(math.sqrt(total_objects))
    intervals = []
    left = mmin
    right = mmin + step
    while True:
        elements_between = list(filter(lambda x: left <= x[0] < right, nums_count_frequency))
        if left > mmax:
            break
        intervals.append((left, right, sum(x[1] for x in elements_between)))
        left = right
        right += step
    intervals = list(map(lambda x: (x[0], x[1], x[2], x[2] / total_objects), intervals))
    return intervals


def print_nums_count_frequency(nums_count_frequency: list[tuple[float, int, float]]) -> None:
    print("DISCRETE SEQUENCE")
    for i in nums_count_frequency:
        print(f'\t{i[0]}\t{i[1]}\t{i[2]}')


def discrete_sequence_from_interval_count(intervals: list[tuple[float, float, int]]) -> list[tuple[float, int]]:
    discrete = []
    for i in intervals:
        discrete.append(((i[0] + i[1]) / 2, i[2]))
    return discrete


def discrete_sequence_from_interval_count_frequency(intervals: list[tuple[float, float, int, float]]) -> list[
    tuple[float, int, float]]:
    discrete = []
    for i in intervals:
        discrete.append(((i[0] + i[1]) / 2, i[2], i[3]))
    return discrete


def x_b_from_nums_count(nums_count: list[tuple[float, int]]) -> float:
    total = 0
    count = 0
    for i in nums_count:
        total += i[0] * i[1]
        count += i[1]
    total /= count
    return total


def s2_from_nums_count(nums_count: list[tuple[float, int]]) -> tuple[float, float]:
    xb = x_b_from_nums_count(nums_count)
    total = 0
    count = 0
    for i in nums_count:
        total += math.pow((xb - i[0]), 2) * i[1]
        count += i[1]
    count = count - 1
    total /= count
    sigma = math.sqrt(total)
    return total, sigma