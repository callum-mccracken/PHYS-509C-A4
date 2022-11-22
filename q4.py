"""Q4 -- Noise"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

N = 10
sum_cos = 0
for m in range(int(N/2)):
    for k in range(N-1):
        print(np.cos(m*k*2*np.pi/N))


