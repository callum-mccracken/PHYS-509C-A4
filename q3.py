"""
Q3 -- Omega Lambda

NOTE: "constant" here = L_0 H_0^2
"""
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import minimize
import utils

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# sigma on m from question
SIGMA_M = 0.1

# open file, get data
filedata = np.loadtxt("data/sn_data.txt")
z = filedata[:,0]
m = filedata[:,1]

# put the data points on a plot
plt.scatter(z, m)

def mean_m(z_i, constant, omega_lambda):
    """For z, constant, and omega_lambda values, return the expected m."""
    return -2.5*np.log10(
        constant/((z_i + 1/2*z_i**2 * ((1 + 3*omega_lambda)/2))**2))

def neg_ll(constant, omega_lembda):
    """Negative log likelihood, derived in text."""
    # Let L_0H_0^2 = "constant"
    mean = mean_m(z, constant, omega_lembda)
    return np.sum(
        np.log(np.sqrt(2*np.pi) * SIGMA_M) + ((m - mean)**2/(2*SIGMA_M**2)))

# the guesses come from asking friends what it should roughly work out to
guess = [4e-16, 7e-01]

# find the precise values by minimizing log likelihood
constant_0, omega_lambda_0 = minimize(
    utils.one_param(neg_ll), guess, method="Nelder-Mead",
    bounds=((0, None), (0, 1))).x
print(f"{constant_0=}, {omega_lambda_0=}")

# plot best-fit curve
z_sorted = np.array(sorted(z))
m_fit = mean_m(z_sorted, constant_0, omega_lambda_0)
plt.plot(z_sorted, m_fit)

# finish up the plot
plt.xlabel("z")
plt.ylabel("m")
plt.tight_layout()
plt.savefig("images/q3_a_fit.png")
plt.cla()
plt.clf()

def neg_ll_one_param(omega_lambda):
    """Negative log likelihood to use with best-fit "constant" value"""
    return neg_ll(constant_0, omega_lambda)

def ll_sigma_bounds(x_vals, neg_log_l_vals, n_sigma):
    """
    Find the uncertainty bounds for a neg log likelihood function.

    This assumes that the uncertainty bounds occur at two the x values given.

    (i.e. make your x as fine-grained as needed, we don't interpolate)
    """
    # find the minimum
    min_y_index = np.where(neg_log_l_vals == min(neg_log_l_vals))[0]
    min_y = neg_log_l_vals[min_y_index]

    if n_sigma == 1:
        increment = 0.5
    else:
        raise ValueError("This bit has not been coded yet")

    thresh = min_y + increment

    # find where function crosses thresh
    first_index = np.where(neg_log_l_vals < thresh)[0][0]
    last_index = np.where(neg_log_l_vals < thresh)[-1][-1]

    # return min, max x bounds
    return x_vals[first_index], x_vals[last_index]

x = np.linspace(0.6, 0.8, 1000)
y = np.array([neg_ll_one_param(xi) for xi in x])
one_sigma_bound_min, one_sigma_bound_max = ll_sigma_bounds(x, y, n_sigma=1)
plt.axvline(x[np.where(y==min(y))], linestyle='--')
plt.axhline(min(y) + 0.5, linestyle='--')
plt.plot(x, y)
plt.xlabel("$\\Omega_\\Lambda$")
plt.ylabel("$-\\ln(L)$")
plt.savefig("images/q3_a_uncert.png")
plt.cla()
plt.clf()

print(
    "central value:", omega_lambda_0,
    "upper uncertainty:", one_sigma_bound_max - omega_lambda_0,
    "lower uncertainty:", omega_lambda_0 - one_sigma_bound_min)

# PART B
# Now do it again, with L_0 = L_0(1 + az)!
# Q = L_0 H_0^2, so Q = Q(1+az) now
print("Starting part B, with L_0 = L_0(1+az)")

def new_mean_m(z_i, constant, omega_lambda, a_value):
    """For a given z, Q, and Ω_Λ value, return the expected (mean) m."""
    return -2.5*np.log10(
        constant*(1+a_value*z_i)/((z_i + 1/2*z_i**2 * (
            (1 + 3*omega_lambda)/2))**2))

def a_prior(a_value):
    """The prior distribution of the variable a -- a Gaussian."""
    return utils.gaussian_pdf(a_value, mu=0, sigma=0.2)

def new_neg_ll(constant, omega_lambda, a_value):
    """Negative log likelihood, derived in text."""
    # Let L_0H_0^2 = Q
    mean = new_mean_m(z, constant, omega_lambda, a_value)
    return np.sum(
        np.log(np.sqrt(2*np.pi) * SIGMA_M) + ((m - mean)**2/(2*SIGMA_M**2))
        ) + np.log(a_prior(a_value))

# the guesses come from asking friends what it should roughly work out to
guess = [4e-16, 7e-01, 0]

# find the precise values by minimizing log likelihood
constant_0, omega_lambda_0, a_0 = minimize(
    utils.one_param(new_neg_ll), guess, method="Nelder-Mead",
    bounds=((0, None), (0, 1), (None, None))).x
print(f"{constant_0=}, {omega_lambda_0=}, {a_0=}")

# plot best-fit curve
z_sorted = np.array(sorted(z))
m_fit = new_mean_m(z_sorted, constant_0, omega_lambda_0, a_0)
plt.scatter(z, m)
plt.plot(z_sorted, m_fit)

# finish up the plot
plt.xlabel("$z$")
plt.ylabel("$m$ (with systematic $a$)")
plt.tight_layout()
plt.savefig("images/q3_b_fit.png")
plt.cla()
plt.clf()

def new_neg_ll_one_param(omega_lambda):
    """Neg LL with only one parameter, for plotting purposes."""
    return new_neg_ll(constant_0, omega_lambda, a_0)

y = np.array([new_neg_ll_one_param(xi) for xi in x])
one_sigma_bound_min, one_sigma_bound_max = ll_sigma_bounds(x, y, n_sigma=1)
plt.axvline(x[np.where(y==min(y))], linestyle='--')
plt.axhline(min(y) + 0.5, linestyle='--')
plt.plot(x, y)
plt.xlabel("$\\Omega_\\Lambda$")
plt.ylabel("$-\\ln(L)$ with systematic $a$")
plt.savefig("images/q3_b_uncert.png")
plt.cla()
plt.clf()

print(
    "central value:", omega_lambda_0,
    "upper uncertainty:", one_sigma_bound_max - omega_lambda_0,
    "lower uncertainty:", omega_lambda_0 - one_sigma_bound_min)
