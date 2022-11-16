"""Q2 -- Age of the Universe"""
import sympy

# define symbols
O, sigma_O, H, sigma_H, sigma_f = sympy.symbols(
    "O, sigma_O, H, sigma_H, sigma_f", real=True)

# define values to plug in later
values = {
    O: 0.3166,  # Omega_m
    sigma_O: 0.0084,
    H: 2.18007223e-18,  # H_0, converted to 1/s units
    sigma_H: 1.94446757e-20,
    sigma_f: 9.87066385e-57  # uncertainty given in question, in 1/s^3 units
}

# define f = O H^3 and get its derivatives to use in the error prop equation
f = O * H**3
df_dH = sympy.diff(f, H)
df_dO = sympy.diff(f, O)

# age of universe and derivatives, for the same purpose
t_u: sympy.Symbol = 2/(3*H*sympy.sqrt(1-O)) * sympy.asinh(sympy.sqrt((1-O)/O))
dt_dH = sympy.diff(t_u, H)
dt_dO = sympy.diff(t_u, O)

# rearrange error propagation equation to get correlation coefficient
rho: sympy.Symbol = (
    sigma_f**2 - df_dH**2 * sigma_H**2 - df_dO**2 * sigma_O**2
    )/(2*df_dH*df_dO*sigma_H*sigma_O)
print(f"Correlation Coefficient: {rho.subs(values):.3g}")

# get error on age of universe
sigma_t = sympy.sqrt(
    rho * 2*dt_dH*dt_dO*sigma_H*sigma_O +
    dt_dH**2 * sigma_H**2 + dt_dO**2 * sigma_O**2)
print(f"t_u: {t_u.subs(values):.3g} +/- {sigma_t.subs(values):.3g}")

# convert to billions of years
GIGAYEARS_PER_SECOND = 3.16887646e-17
gyrs = t_u.subs(values) * GIGAYEARS_PER_SECOND
gyrs_uncert = sigma_t.subs(values) * GIGAYEARS_PER_SECOND
print(f"Or in billion years, {gyrs:.3f} +/- {gyrs_uncert:.3f}")
