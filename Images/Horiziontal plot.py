# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:34:28 2022

@author: ismae
"""

from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import cmath
from scipy.stats import chisquare as cs

def distr(x,A,sx,x0):
    return A/(sx)*np.exp(-0.5*(x-x0)**2/sx**2)
def distr2(x,A,sx,x0):
    return A*(np.cos((x-x0)*sx))**4
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
f="Horiz Values leer.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
ymin = np.amin(data[:,1])
P0=[100, 2,8]
print(data[:,0])
#data[:,0] = (data[:,0]-xmax+10)*2e-3
#data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
print("p=",p)
print("cov=", np.diag(cov)**0.5)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Avg counts")
ax.plot(data[:,0]-xmax,data[:,1], "ko", label="Measurement")
ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Gaussian Fit")
xplt1=np.linspace(0,p[1], 2)
ax.plot(xplt1, xplt1*0 +distr(p[1],*p[0:2],0), "k-|")
# ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
ax.text(-0.5, 225, "$\sigma \\approx $"+str("%.2f"%p[1])+" mm", fontsize="large",color="k",backgroundcolor="white")
plt.legend(loc=1)
plt.savefig('Horiz.eps', format='eps')
print(2*p[1]*1e-3/(1.9+1.50+1.46/2))
# P0=[277.40655667,1.9e-1,7.90313202]
# #plt.savefig(controlfits+foldername[k] +'_line_' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z))+'_fit.png')
# p,cov=fit(distr2,data[2:6,0],data[2:6,1],p0=P0)
# print("p=",p)
# print("cov=", np.diag(cov)**0.5)
# xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
# xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
# ax.set_xlabel("mm")
# ax.set_ylabel("Avg counts")
# ax.plot(data[:,0]-xmax,data[:,1], "go", label="Measurement")
# ax.plot(xplt-xmax, distr2(xplt,*p), "r-", label="Gaussian Fit")
# xplt1=np.linspace(0,p[1], 100)
# ax.plot(xplt1, xplt1*0)
# plt.legend(loc=1)