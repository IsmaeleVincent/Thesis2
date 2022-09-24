#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:18:39 2022

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:28:47 2022

@author: aaa
"""




from scipy.special import erfc
from scipy.stats import exponnorm
from scipy.stats import norm
from scipy.optimize import curve_fit as fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import inspect, os, time
def rho1(x, B, A, x0, sk, sig):
    g = exponnorm(loc=x0, K=sk, scale=sig)
    return B+A*g.pdf(x)
def rho2(x, B, A, x0, sk, sig):
    g = norm(loc=x0, scale=sig)
    return B+A*g.pdf(x)
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]

mu = 1.5e-3
tau = 0.1
sigma = 1.2e-3


x = exponnorm.ppf(np.arange(0.01, 0.992, 1e-2), K=tau, loc=mu, scale=sigma)
print("x[0] = ",x[0])
plt.plot(x, rho1(x, 0, 1, mu, tau, sigma), "k.-")
# plt.plot(x, rho2(x, 0, 1, mu,0, sigma), "k.-")
Dx = x[1:]-x[:-1]
print("Dx avg=",np.average(Dx),"; Dx min=", np.amin(Dx),"; Dx max=", np.amax(Dx))
