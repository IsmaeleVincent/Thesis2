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
def distr1(x,A, B, x0, sig):
    return A + B*cosine(loc=x0, scale=sig).pdf(x)

fig = plt.figure(figsize=(6,10))
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
f="Values 0deg.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
data[:,1]=data[:,1]/ymax
P0=[0, 100, 6, 5]
P0=[0, 100, 6, 5]
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
p1,cov1=fit(distr1,data[:,0],data[:,1],p0=P0)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Avg counts (arb)")
ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Gaussian Fit")
xplt1=np.linspace(0,p[3], 2)
ax.plot(xplt1, xplt1*0 +distr(p[2]+p[3],*p), "k-|")
ax.text(p[3], distr(p[2]+p[3],*p), "$\sigma_g \\approx $"+str("%.2f"%p[3])+" mm", fontsize="large",color="k")
ax.plot(data[:,0]-xmax,data[:,1], "ko", label="Measurement 0deg")
ax.legend(loc=1)
f="Values 70deg.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
data[:,1]=data[:,1]/ymax
P0=[0, 100, 6, 5]
P0=[0, 100, 6, 5]
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
p1,cov1=fit(distr1,data[:,0],data[:,1],p0=P0)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax1.set_xlabel("mm")
ax1.set_ylabel("Avg counts (arb)")
ax1.plot(xplt-xmax, distr(xplt,*p), "k--", label="Gaussian Fit")
xplt1=np.linspace(0,p[3], 2)
ax1.plot(xplt1, xplt1*0 +distr(p[2]+p[3],*p), "k-|")
ax1.text(p[3], distr(p[2]+p[3],*p), "$\sigma_g \\approx $"+str("%.2f"%p[3])+" mm", fontsize="large",color="k")
ax1.plot(data[:,0]-xmax,data[:,1], "ko", label="Measurement 70deg")
ax1.legend(loc=1)
# plt.savefig('/home/aaa/Desktop/Thesis2/Images/Horiz.eps', format='pdf')
