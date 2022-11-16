"""Q1 -- S%P 500"""

import random
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import utils
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# load data, use datetime module for dates
file_data = np.loadtxt("data/stockdata.txt", dtype="str")
dates = file_data[:,0]
values = np.array(file_data[:,1], dtype=np.float32)
datetimes = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])

# plot the value over time (not required, just nice)
plt.plot(datetimes, values)
plt.xlabel("Year")
plt.ylabel("Value of S&P 500")
plt.savefig("images/q1_value_over_time.png")
plt.cla()
plt.clf()

# calculate returns, will have length 1 less than that of values
returns = np.array([
    (values[i] - values[i-1])/values[i] for i in range(1, len(values))])

# plot the data in a histogram, density=True ensures histogram is normalized
plt.hist(returns, bins=100, density=True, label="Data")

def gauss_neg_ll(mean, sigma):
    """See pdf for how we got this, negative log likelihood for Gaussian."""
    return np.sum(
        np.log(np.sqrt(2*np.pi)*sigma) + (returns - mean)**2 / (2*sigma**2))

# got this guess from looking at the histogram
gauss_guess = (0.0001, 0.01)
# minimize log likelihood to get gaussian params
mu_0, sigma_0 = minimize(
    utils.one_param(gauss_neg_ll), gauss_guess, method="Nelder-Mead").x
print(f"{mu_0=}, {sigma_0=}")

# add the Gaussian to the plot
r = sorted(returns)
gaussian_fit_pdf = utils.gaussian_pdf(r, mu=mu_0, sigma=sigma_0)
plt.plot(r, gaussian_fit_pdf, label="Gaussian ML best-fit distribution")


def laplace_neg_ll(peak, width):
    """
    See pdf for how we got this, negative log likelihood for Laplace.

    peak = A, width = B
    """
    return np.sum(
        np.log(2*np.abs(width)) + np.abs(returns - peak) / np.abs(width))


# use another guess, minimize Laplace distribution
laplace_guess = (0.0001, 0.01)
A_0, B_0 = minimize(
    utils.one_param(laplace_neg_ll), laplace_guess, method="Nelder-Mead").x
print(f"{A_0=}, {B_0=}")


def laplace_pdf(r_val, peak, width):
    """Laplace PDF as given in the question, r_val = R, peak=A, width=B"""
    return 1/(2*width) * np.exp(-np.abs(r_val-peak)/width)

# add the Laplace distribution to the plot
plt.plot(r, laplace_pdf(r, A_0, B_0), label="Laplace ML best-fit distribution")

# plot everything, using a log scale like the question asks for
plt.yscale("log")
plt.xlabel("Return")
plt.ylabel("log(P(R))")
plt.legend()
plt.savefig("images/q1_return_hist.png")
plt.cla()
plt.clf()

# now generate random variables from each distribution
N_DAYS = 250*30
N_TRIALS = 1000

# arrays to store results at the end of each trial
gaussian_results = np.empty(N_TRIALS)
laplace_results = np.empty(N_TRIALS)
data_results = np.empty(N_TRIALS)

# arrays to store simulation values during each trial
gaussian_values = np.empty(N_DAYS)
laplace_values = np.empty(N_DAYS)
data_values = np.empty(N_DAYS)

print("Simulating... 🤓")
for trial in tqdm(range(N_TRIALS)):
    # start from 100
    gaussian_values[0] = 100
    laplace_values[0] = 100
    data_values[0] = 100

    # calculate return for each day (we'll ignore the first day)
    gaussian_returns = np.random.normal(mu_0, sigma_0, N_DAYS)
    laplace_returns = np.random.laplace(A_0, B_0, N_DAYS)
    data_returns = random.sample(sorted(returns), N_DAYS)

    # calculate subsequent values
    for day in range(N_DAYS-1):
        gaussian_values[day+1] = gaussian_values[day] / (
            1-gaussian_returns[day+1])
        laplace_values[day+1] = laplace_values[day] / (
            1-laplace_returns[day+1])
        data_values[day+1] = data_values[day] / (
            1-data_returns[day+1])

    # store final results
    gaussian_results[trial] = gaussian_values[-1]
    laplace_results[trial] = laplace_values[-1]
    data_results[trial] = data_values[-1]

# plot histograms
plt.hist(gaussian_results, label="Gaussian", bins=50, alpha=0.5, density=True)
plt.hist(laplace_results, label="Laplace", bins=50, alpha=0.5, density=True)
plt.hist(data_results, label="Data", bins=50, alpha=0.5, density=True)
plt.xlabel("Value v After 30 Years (truncated at 30k)")
plt.ylabel("P(v)")
plt.legend()
plt.xlim(0, 30000)
plt.savefig("images/q1_generator_comparison.png")
plt.cla()
plt.clf()
