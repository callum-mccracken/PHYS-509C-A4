"""Q3 -- FC Confidence Intervals"""
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.integrate import quad
from tqdm import tqdm
import utils

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def likelihood(k, p, N):
    """Binomial function, normalized"""
    return utils.binomial_pmf(k, N, p)

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

k = np.arange(11)
upper_limits = np.array([-np.inf]*len(k))
lower_limits = np.array([np.inf]*len(k))

for p in tqdm(np.linspace(0, 1, 1000)):
    # Highest R will be when p=k/N
    likelihoods = likelihood(k, p, N=10)
    ratios = ratio(k, p, N=10)

    # start from the highest R
    sort_order = np.array(list(reversed(np.argsort(ratios))))
    likelihoods = likelihoods[sort_order]
    ratios = ratios[sort_order]
    k = k[sort_order]
    integral = 0
    index = 0
    lower_k = np.inf
    upper_k = -np.inf
    while integral <= 0.9:
        integral += likelihoods[index]
        if k[index] < lower_k:
            lower_k = k[index]
        if k[index] > upper_k:
            upper_k = k[index]
        index += 1

    # update upper/lower limits at the bounds we found
    if p > upper_limits[lower_k]:
        upper_limits[lower_k] = p
    if p < lower_limits[upper_k]:
        lower_limits[upper_k] = p

# we only want the last one
print(lower_limits[-1], upper_limits[-1])


k = k[np.argsort(k)]
print(k)
#upper_limits = upper_limits[sort_order]
#lower_limits = lower_limits[sort_order]
plt.step(k, upper_limits, label="Upper limit")
plt.step(k, lower_limits, label="Lower limit")
plt.legend()
plt.show()
