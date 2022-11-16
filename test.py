"""
Assume that the data follows an exponential plateau,
approaching some steady state value C.

Do a maximum likelihood fit for this level,
and determine the uncertainty on it.

Note that you have not been given the uncertainties on the measured
values -- instead, assume that all measurements have the same uncertainty
level, and fit for it as one of the parameters in your fit.

Submit your code, the functional form you fit,
and your result for the steady state value (with uncertainty).

What uncertainty value per data point did you get?

How much of that uncertainty can be attributed to the time binning
(measurements are only reported to the nearest minute)?
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats

# t = minutes after start time
t_data = np.array([0, 6, 10, 14, 19, 24, 29, 33, 46, 54, 57])
# y = ppm
y_data = np.array([484, 501, 520, 535, 554, 565, 579, 593, 635, 651, 654])

def fit_function(t,C,B,A):
    return C - B*np.exp(-A*t)

trials = np.array(range(1000))
for trial in trials:
    
    def neg_ll(params):
        C, B, A, sigma_y = params
        y_pred = fit_function(t_data, C, B, A)
        return -np.sum(stats.norm.logpdf(y_data, loc=y_pred, scale=sigma_y))

    guess = [900, 300, 0.01, 10]
    results = minimize(neg_ll, guess, method='Nelder-Mead')

    C_0, B_0, A_0, sigma_0 = results.x
    print(C_0, B_0, A_0, sigma_0)
    plt.plot(t_data, y_data, 'go')
    plt.errorbar(t_data, fit_function(t_data, C_0, B_0, A_0), yerr=sigma_0)
    plt.savefig("images/q3.png")

    # How much of that uncertainty can be attributed to the time binning
    # (measurements are only reported to the nearest minute)?

