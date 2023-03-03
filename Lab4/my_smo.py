import numpy as np
import math


# https://www.symbolab.com/solver/system-of-equations-calculator
def kolmogorov_graph(matrix: list[list[float]]):
    """
        See practice 8, variant 1 - 5, exercise 4,
        picture (8.9)
        in rows - direction out of the node
        in columns - direction into the node
        matrix = [
               1  2  3  4
            1 [0, 2, 6, 7],
            2 [0, 0, 3, 0],
            3 [0, 0, 0, 0],
            4 [5, 6, 5, 0]
        ]
        1 is going out to 2 with 2
        1 is going out to 3 with 6
        1 is going out to 4 with 7
        2 is going out to 3 with 3
        4 is going out to 1 with 5
        4 is going out to 2 with 6
        4 is going out to 3 with 5
    """
    size = len(matrix)
    coefficients = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            coefficients[i][i] -= matrix[i][j]
        for j in range(size):
            coefficients[i][j] += matrix[j][i]
    equations = []
    for i in range(size):
        eq = ''
        for j in range(size):
            eq += f' + ({coefficients[i][j]}) * p_{j + 1}'
        eq += ' = 0'
        equations.append(eq)
    eq = ''
    for i in range(size):
        eq += f'+ p_{i + 1} '
    eq += '= 1'
    equations.append(eq)
    s = format(equations)
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace("'", '')
    print(s)


def regular_stream(l: list[tuple[float, float]]):
    """
        See practice 8, example 8.3
        l = [
            (1, 5),
            (2, 4)
        ]
    """
    n = len(l)
    coefficients = [1]
    for i, el in enumerate(l):
        coef = el[0] / el[1]
        prev_el = coefficients[i]
        coefficients.append(prev_el * coef)
    p_0 = math.pow(sum(coefficients), -1)
    coef_ps = [p_0]
    for el in coefficients[1:]:
        coef_ps.append(p_0 * el)
    for i, el in enumerate(coef_ps):
        print(f'p_{i} = {el}')


def cyclic_stream(l: list[float]):
    """
        See practice 8, example 8.3, image 8.7
        l = [2, 3, 4, 5]
    """
    inverted = [1 / x for x in l[1:]]
    p_0 = math.pow(1 + l[0] * sum(inverted), -1)
    coefs_ps = [p_0]
    for el in inverted:
        coefs_ps.append(p_0 * el)
    print(f'cyclic_coefs: {coefs_ps}')
    for i, el in enumerate(coefs_ps):
        print(f'p_{i} = {el}')


def matrix_transition(l: list[float], matrix: list[list[float]]):
    """
        See example 8.4
        l = [0.6, 0.4, 0]
        matrix = [
            [0.5, 0.4, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.4, 0.4]
        ]
    """
    l_arr = np.array(l)
    mat_arr = np.array(matrix)
    result = np.matmul(l_arr, mat_arr)
    print(format(result).replace(' ', ', '))
    return result


def single_channel_with_cancelling(lam: float, mu: float):
    Q = mu / (lam + mu)
    A = lam * Q
    p_cancelling = lam / (lam + mu)
    print('SINGLE-CHANNEL WITH CANCELLING')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')


def multichannel_with_cancelling(lam: float, mu: float, n: int):
    ro = lam / mu
    p_0 = 1 / (1 + sum([math.pow(ro, i) / math.factorial(i) for i in range(1, n + 1)]))
    p_list = [p_0]
    p_list.extend([math.pow(ro, i) * p_0 / math.factorial(i) for i in range(1, n + 1)])
    p_n = p_list[-1]
    Q = 1 - p_n
    A = lam * (1 - p_n)
    K_mean = A / mu
    p_cancelling = p_n
    print('MULTI-CHANNEL WITH CANCELLING')
    print(f'n a.k.a кількість каналів: {n}')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    for i, el in enumerate(p_list):
        print(f'p_{i}: {el}')

# http://manualsem.com/book/766-imovirnisni-procesi/25-odnokanalna-sistema-masovogo-obslugovuvannya-z-obmezhenoyu-chergoyu.html
def single_channel_with_limited_queue(lam: float, mu: float, m: int):
    ro = lam / mu
    q_0 = (1 - ro) / (1 - math.pow(ro, m + 2))
    p_list = [q_0]
    p_list.extend([math.pow(ro, i) * q_0 for i in range(1, m + 2)])
    q_m_plus_one = p_list[-1]
    A = lam * (1 - q_m_plus_one)
    Q = A / lam
    p_cancelling = q_m_plus_one
    K_mean = 1 - q_0
    r_mean_nominator = math.pow(ro, 2) * (1 - math.pow(ro, m) * (m + 1 - m * ro))
    r_mean_denominator = (1 - math.pow(ro, m + 2)) * (1 - ro)
    r_mean = r_mean_nominator / r_mean_denominator
    z_mean = K_mean + r_mean
    t_mean = z_mean / lam
    t_queue_mean = r_mean / lam
    print('SINGLE-CHANNEL WITH LIMITED QUEUE')
    print(f'm a.k.a розмір черги: {m}')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    print(f'z_mean a.k.a середнє число заявок у СМО ( тобто всі заявки, які обслоговуються та чекають у черзі, якщо вона існує ) {z_mean}')
    print(f'r_mean a.k.a середнє число заявок в черзі {r_mean}')
    print(f't_mean a.k.a середній час перебування в СМО {t_mean}')
    print(f't_queue_men a.k.a середній час перебування заявки в черзі: {t_queue_mean}')
    for i, el in enumerate(p_list):
        print(f'p_{i}: {el}')


# http://manualsem.com/book/766-imovirnisni-procesi/26-odnokanalna-sistema-masovogo-obslugovuvannya-z-neobmezhenoyu-chergoyu.html
def single_channel_with_unlimited_queue(lam: float, mu: float):
    p_cancelling = 0
    Q = 1
    A = lam
    ro = lam / mu
    q_0 = 1 - ro
    K_mean = ro
    r_mean = math.pow(ro, 2) / (1 - ro)
    z_mean = ro / (1 - ro)
    t_mean = ro / (lam * (1 - ro))
    t_queue_mean = math.pow(ro, 2) / (lam * (1 - ro))
    print('SINGLE-CHANNEL WITH UNLIMITED QUEUE')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    print(f'z_mean a.k.a середнє число заявок у СМО ( тобто всі заявки, які обслоговуються та чекають у черзі, якщо вона існує ) {z_mean}')
    print(f'r_mean a.k.a середнє число заявок в черзі {r_mean}')
    print(f't_mean a.k.a середній час перебування в СМО {t_mean}')
    print(f't_queue_men a.k.a середній час перебування заявки в черзі: {t_queue_mean}')
    print(f'q_0: {q_0}')


# http://manualsem.com/book/766-imovirnisni-procesi/28-bagatokanalna-sistema-masovogo-obslugovuvannya-z-neobmezhenoyu-chergoyu.html
def multichannel_with_unlimited_queue(lam: float, mu: float, n: int):
    ro = lam / mu
    X = ro / n
    if X >= 1:
        print('Фінальний розподіл не може існувати')
        return
    ro_sum = sum([math.pow(ro, i) / math.factorial(i) for i in range(1, n + 1)])
    q_0 = 1 / (1 + ro_sum + math.pow(ro, n + 1) / ((1 - X) * n * math.factorial(n)))
    r_mean = math.pow(ro, n + 1) * q_0 / (n * math.pow(1 - X, 2) * math.factorial(n))
    A = ro
    K_mean = ro
    Q = 1
    p_cancelling = 0
    z_mean = r_mean + K_mean
    t_mean = z_mean / lam
    t_queue_mean = r_mean / lam
    print('MULTI-CHANNEL WITH UNLIMITED QUEUE')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'X: {X}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    print(f'z_mean a.k.a середнє число заявок у СМО ( тобто всі заявки, які обслоговуються та чекають у черзі, якщо вона існує ) {z_mean}')
    print(f'r_mean a.k.a середнє число заявок в черзі {r_mean}')
    print(f't_mean a.k.a середній час перебування в СМО {t_mean}')
    print(f't_queue_men a.k.a середній час перебування заявки в черзі: {t_queue_mean}')
    print(f'q_0: {q_0}')


# коли є слово "розподілена за довільним законом", то використовуй це
def single_channel_arbitrary_with_unlimited_queue(lam: float, mu: float, deviation: float):
    M = 1 / mu
    ro = lam / mu
    v = deviation * mu
    r_mean = math.pow(ro, 2) * (1 + math.pow(v, 2)) / (2 * (1 - ro))
    t_queue_mean = r_mean / lam
    K_mean = ro
    z_mean = r_mean + K_mean
    t_mean = t_queue_mean + 1 / mu
    print(f'Mean: {M}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    print(f'z_mean a.k.a середнє число заявок у СМО ( тобто всі заявки, які обслоговуються та чекають у черзі, якщо вона існує ) {z_mean}')
    print(f'r_mean a.k.a середнє число заявок в черзі {r_mean}')
    print(f't_mean a.k.a середній час перебування в СМО {t_mean}')
    print(f't_queue_men a.k.a середній час перебування заявки в черзі: {t_queue_mean}')


# http://manualsem.com/book/766-imovirnisni-procesi/27-bagatokanalna-sistema-masovogo-obslugovuvannya-z-obmezhenoyu-chergoyu.html
def multichannel_with_limited_queue(lam: float, mu: float, n: int, m: int):
    ro = lam / mu
    q_0 = 1 + sum([math.pow(ro, i) / math.factorial(i) for i in range(1, n)])
    ro_n = (math.pow(ro, n) / math.factorial(n))
    q_0 += ro_n * (1 + sum([math.pow(ro, i) / math.pow(n, i) for i in range(1, m + 1)]))
    q_0 = 1 / q_0
    p_list = [q_0]
    p_list.extend([math.pow(ro, i) * q_0 / math.factorial(i) for i in range(1, n + 1)])
    q_n = p_list[-1]
    p_list.extend([q_n * math.pow(ro, i) / math.pow(n, i) for i in range(1, m + 1)])
    p_cancelling = p_list[-1]
    Q = 1 - p_cancelling
    A = lam * Q
    X = ro / n
    r_mean = math.pow(ro, n + 1) * q_0 * (1 - math.pow(X, m) * (m + 1 - m * X))
    r_mean /= n * math.factorial(n) * math.pow(1 - X, 2)
    z_mean = A / mu
    K_mean = r_mean + z_mean
    t_queue_mean = r_mean / lam
    t_mean = t_queue_mean + Q / mu
    print('MULTI-CHANNEL WITH LIMITED QUEUE')
    print(f'n a.k.a кількість каналів: {n}')
    print(f'm a.k.a розмір черги: {m}')
    print(f'lamda a.k.a інтенсивність потоку: {lam}')
    print(f'mu a.k.a інтенсивність потоку обслоговування: {mu}')
    print(f'ro a.k.a клефіцієнт завантаження системи: {ro}')
    print(f'X: {X}')
    print(f'A a.k.a середнє число заявок, які обслоговуються за одиницю часу: {A}')
    print(f'Q a.k.a ймовірність обслоговування, або відносна пропускна спроможність: {Q}')
    print(f'p_cancelling a.k.a імовірність відмови: {p_cancelling}')
    print(f'K_mean a.k.a середнє число зайнятих каналів: {K_mean}')
    print(f'z_mean a.k.a середнє число заявок у СМО ( тобто всі заявки, які обслоговуються та чекають у черзі, якщо вона існує ) {z_mean}')
    print(f'r_mean a.k.a середнє число заявок в черзі {r_mean}')
    print(f't_mean a.k.a середній час перебування в СМО {t_mean}')
    print(f't_queue_men a.k.a середній час перебування заявки в черзі: {t_queue_mean}')
    for i, el in enumerate(p_list):
        print(f'p_{i}: {el}')

def prod(l: list):
    result = l[0]
    for el in l[1:]:
        result *= el
    return result

def multichannel_with_limited_time_in_queue(lam: float, mu: float, v: float, n: int, r: int):
    ro = lam / mu
    p_0 = 1 + sum([math.pow(ro, i) / math.factorial(i) for i in range(n)])
    ro_n = math.pow(ro, n) / math.factorial(n)
    p_0 += ro_n * (1 + sum(
        [
            math.pow(lam, i) / prod([n*mu + k*v for k in range(i + 1)]) for i in range(r + 1)
        ]
    ))
    p_list = [p_0]
    p_list.extend([math.pow(ro, i) / math.factorial(i) for i in range(n + 1)])
    p_n = p_list[-1]
    p_list.extend([
        math.pow(lam, i) / prod([n * mu + k * v for k in range(i + 1)]) for i in range(r + 1)
    ])




# kolmogorov_graph(
#     [
#         [0, 2, 6, 7],
#         [0, 0, 3, 0],
#         [0, 0, 0, 0],
#         [5, 6, 5, 0]
#     ]
# )
# regular_stream(
#     [
#         (1, 5),
#         (2, 4)
#     ]
# )
# cyclic_stream([2, 3, 4, 5])
# matrix_transition(
#     [0.24, 0.42, 0.34],
#     [
#         [0.1, 0.5, 0.4],
#         [0.6, 0.2, 0.2],
#         [0.3, 0.4, 0.3]
#     ]
# )
# single_channel_with_cancelling(24 / 15, 0.5)
# multichannel_with_cancelling(0.8, 1 / 3, 5)
# single_channel_with_unlimited_queue(3, 4)
# multichannel_with_unlimited_queue(140/24, 5, 2)
# single_channel_arbitrary_with_unlimited_queue(0.1, 1 / 6, 1)
# single_channel_with_limited_queue(1, 0.8, 3)
# multichannel_with_limited_queue(2, 0.5, 2, 3)



# 2
# kolmogorov_graph([
#     [0, 1, 2, 0],
#     [2, 0, 0, 2],
#     [3, 0, 0, 1],
#     [0, 3, 2, 0],
# ])

# 3
#cyclic_stream([2, 3, 4, 5])

# 5
# matrix_transition(
#     [0.6, 0.4, 0],
#     [[0.5, 0.4, 0.1],
#     [0.3, 0.5, 0.2],
#     [0.2, 0.4, 0.4]]
# )
# [0.42, 0.44, 0.14]
# matrix_transition(
#     [0.42, 0.44, 0.14],
#     [[0.5, 0.4, 0.1],
#     [0.3, 0.5, 0.2],
#     [0.2, 0.4, 0.4]]
# )
# [0.37, , 0.444, 0.186]

# s = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'
# d = '0.5 0.9 1.4 2.5 3.5 3.9 4.5 5.1 7.1 7.8 8.2 8.7 9.6 12.3 13.6 15.4 17.3'
# print(s.replace(' ', ', '))
# print(d.replace(' ', ', '))

