"""Q3 -- FC Confidence Intervals"""
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.integrate import quad
import utils

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def likelihood(k, p, N):
    """Binomial function, normalized"""
    return 11*utils.binomial_pmf(k, N, p)

def ratio(k, p, N):
    """Likelihood ratio using ML best p = k/N."""
    p_best = k/N
    return likelihood(k, p, N) / likelihood(k, p_best, N)

def get_p_bounds(p, R, value):
    """Get the bounds on p such that R > value for p in [left, right]"""
    left_bound, right_bound = None, None
    if R[0] >= value:
        left_bound = p[0]
        for i, r in enumerate(R):
            if r <= value:
                right_bound = p[i]
                return left_bound, right_bound
    else:
        for i, r in enumerate(R):
            if r >= value and left_bound is None:
                left_bound = p[i]
            if r <= value and left_bound is not None:
                right_bound = p[i]
                return left_bound, right_bound
    if right_bound is None:
        right_bound = p[-1]
    return left_bound, right_bound

k = list(range(11))
upper_limits = []
lower_limits = []

for k_i in k:
    print(k_i)
    # Highest R will be when p=k/N
    p = np.linspace(0, 1, 1000)
    R = ratio(k_i, p, N=10)
    # start from the highest R = 1
    integral = 0
    r_limit = 1
    while integral <= 0.9:
        r_limit -= 0.001
        p_bounds = get_p_bounds(p, R, r_limit)
        int_func = lambda p: likelihood(k_i, p, N=10)
        integral = quad(int_func, *p_bounds)[0]
    lower_limits.append(p_bounds[0])
    upper_limits.append(p_bounds[1])

# we only want the last one
print(lower_limits[-1], upper_limits[-1])
plt.step(k, upper_limits, label="Upper limit")
plt.step(k, lower_limits, label="Lower limit")
plt.legend()
plt.show()
