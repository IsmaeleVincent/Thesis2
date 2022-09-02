# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:34:28 2022

@author: ismae
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.stats import norm
from scipy.stats import cosine
from scipy.stats import exponnorm

def distr(x,A, B, x0, sig):
    return A + B*norm(loc=x0, scale=sig).pdf(x)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
f="/home/aaa/Desktop/Thesis2/Images/Horiz Values 48deg.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
ymin = np.amin(data[:,1])
P0=[30, 300, 6, 2]
# data[:,0] = (data[:,0]-xmax+10)*2e-3
# data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
print("p=",p)
print("cov=", np.diag(cov)**0.5)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Avg counts (arb)")
ax.plot(data[:,0]-xmax,data[:,1], "ko", label="Measurement")
ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Gaussian Fit")
xplt1=np.linspace(0,p[3], 2)
ax.plot(xplt1, xplt1*0 +distr(p[2]+p[3],*p), "k-|")
# ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
ax.text(-0.5, distr(p[2]+p[3],*p)+30, "$\sigma \\approx $"+str("%.2f"%p[3])+" mm", fontsize="large",color="k",backgroundcolor="white")
plt.legend(loc=1)
plt.savefig('/home/aaa/Desktop/Thesis2/Images/Horiz2.eps', format='pdf')
