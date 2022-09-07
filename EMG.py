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

import numpy as np
import inspect,os,time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit as fit
from scipy.stats import skewnorm
from scipy.stats import exponnorm
from scipy.special import erfc

def rho1(x,B,A,x0, sk, sig):
    g=exponnorm(loc=x0,K=sk,scale=sig)
    return B+A*g.pdf(x)

tau=10
mu=2e-3
sigma=0.2e-3
x=exponnorm.ppf(np.arange(0.01,0.99,1e-2),K=tau, loc=mu, scale=sigma)
plt.plot(x, rho1(x,0,1,mu,tau,sigma))