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

plt.rcParams['font.size'] = 15
# plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
f="/home/aaa/Desktop/Thesis2/Images/Horiz Values 0deg.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
ymin = np.amin(data[:,1])
P0=[30, 500, 6, 2]
# data[:,0] = (data[:,0]-xmax+10)*2e-3
# data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
print("p=",p)
print("cov=", np.diag(cov)**0.5)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Avg counts")
ax.plot(data[:,0]-xmax,data[:,1], "ko", label="Measurement")
ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Gaussian Fit")
xplt1=np.linspace(0,p[3], 2)
ax.plot(xplt1, xplt1*0 +distr(p[2]+p[3],*p), "k-|", markersize=10)
# ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
ax.text(-0.5, distr(p[2]+p[3],*p)+22, "$\sigma \\approx $"+str("%.2f"%p[3])+" mm", color="k",backgroundcolor="white")
plt.legend(loc=1, framealpha=1, fontsize=14)
plt.savefig('/home/aaa/Desktop/Thesis2/Images/Horiz.eps', format='pdf')
