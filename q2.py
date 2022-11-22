"""Q2 -- Chi-squared fits with systematics"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def main():
    """Do some chi-squared fitting."""

    # Part A:
    # Theory: y = 3x^2 -1.
    # A dataset is obtained.
    # The resolution on each y measurement is 0.02.
    # Use a chi^2 statistic to test whether the theory.
    # Quote a p-value.

    # dataset
    x = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.array([
        -0.951, -0.842, -0.741, -0.492,
        -0.229, 0.118, 0.494, 0.957, 1.449, 2.055])
    sigma_y = 0.02

    # model values
    expected_y = 3*x**2 - 1

    # cheeky lil plot
    plt.plot(x, y, 'k.')
    plt.plot(x, expected_y, 'b--')
    plt.savefig("q2_data_and_expected.png")
    plt.cla()
    plt.clf()

    # calculate chi2 and p-value
    chi_squared_value = np.sum(((y - expected_y)/sigma_y)**2)
    n_dof = len(y)
    p_value = 1 - stats.chi2.cdf(chi_squared_value, df=n_dof)
    print(f"{chi_squared_value=:.5f}, {p_value=:.5f}")

    # then part b):
    # Add a systematic: dy=ax --> y = y+ax, a=const. a = 0 +- 0.05
    # Get chi2 and p-value again.

    sigma_a = 0.05

    # first make covariance matrix
    cov_mtx = np.empty((len(x),len(x)))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            delta_ij = int(i == j)
            cov_mtx[i, j] = x_i*x_j*sigma_a**2 + delta_ij*sigma_y**2

    inv_cov_mtx = np.linalg.inv(cov_mtx)

    # Then calculate the chi^2 thing
    # Use a=0 in f(x_i|a) since we know a=0 is the best-fit value
    chi2_with_syst = 0
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            front = (y[i] - (3*x_i**2 - 1))
            back = (y[j] - (3*x_j**2 - 1))
            chi2_with_syst += front * inv_cov_mtx[i,j] * back

    n_dof_with_syst = len(y) - 1
    p_value_with_syst = 1 - stats.chi2.cdf(chi2_with_syst, df=n_dof_with_syst)
    print(f"{chi2_with_syst=:.5f}, {p_value_with_syst=:.5f}")



if __name__ == "__main__":
    main()
