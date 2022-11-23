"""Q1 -- Medical Trials"""

import numpy as np
from scipy import stats
from scipy.stats import binomtest
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 100

def chi_squared_expression(n):
    """Return the chi^2 thing we derived in the text."""
    return 2*(N * np.log(2) + n*np.log(n/N) + (N-n)*np.log(1-n/N))

# start at N/2 since we don't want to get extra-small values near zero
for n in range(0, N+1):
    chi2 = chi_squared_expression(n)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    if p_value < 0.05:
        print(p_value, f'happened at {chi2=}, {n=}')

n = np.array(list(range(N+1)))
chi2 = chi_squared_expression(n)
p_value = 1 - stats.chi2.cdf(chi2, df=1)
plt.plot(n, p_value)
plt.axhline(0.05)
plt.xlabel("Number of survivors in 100 trials")
plt.ylabel("p-value")
plt.savefig("q1_a.png")

# from the print lines and such, found this:
n_part_a = 60

# Simulate the more complex scenario in part B:
def simulate(n_trials=100):
    n_99cl = 0
    n_95cl = 0

    # assuming p=0.5 since null hypothesis is true
    p = 0.5

    for _ in tqdm(range(n_trials)):
        # to check if we're done this trial
        finished = False

        # start with 0-24 = random trials, binomial
        # p=0.5 so we can use this randint(2) thing
        n_survived = np.random.binomial(25, 0.5)

        # at n=25-99 decide if we should quit (i.e. 99% CL success)
        for N in range(25,100):
            # this is the sum of P(n_i|N,p) for n_i >= n_survived
            p_recover_by_chance = binomtest(
                k=n_survived, n=N, p=p, alternative="greater").pvalue
            if p_recover_by_chance < 0.01:
                n_99cl += 1
                finished = True
                break
            # decide if the next patient lives or dies
            n_survived += np.random.binomial(1, 0.5)
        if not finished:
            # at n=100
            if n_survived > n_part_a:
                n_95cl += 1

    p_reject_null = (n_99cl + n_95cl)/n_trials
    p_99cl = n_99cl/n_trials
    return p_reject_null, p_99cl

print(simulate(100000))
