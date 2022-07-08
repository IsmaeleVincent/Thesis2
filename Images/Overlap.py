# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:34:28 2022

@author: ismae
48 line 70 theta 9
"""

from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import cmath
from scipy.stats import chisquare as cs

def distr(x,A1,sx1,x01,A2,sx2,x02,A3,sx3,x03):
    return A1/(sx1)*np.exp(-0.5*(x-x01)**2/sx1**2)+A2/(sx2)*np.exp(-0.5*(x-x02)**2/sx2**2)+A3/(sx3)*np.exp(-0.5*(x-x03)**2/sx3**2)

def gauss(x,A,sx,x0):
    return A/(sx)*np.exp(-0.5*(x-x0)**2/sx**2)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
# f="Overlap.csv"
# data = np.loadtxt(f, skiprows=1, delimiter=",")
# data[:,0]*=2
# ymax = np.amax(data[:,1])
# ymin = np.amin(data[:,1])
# P0=[1000, 2,0, 600, 2, 20, 100, 3,30]
# print(data[:,0])
# #data[:,0] = (data[:,0]-xmax+10)*2e-3
# #data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
# p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
# print("p=",p)
# print("cov=", np.diag(cov)**0.5)
# xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
# xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
# ax.set_xlabel("mm")
# ax.set_ylabel("Counts")
# ax.plot(xplt-xmax, gauss(xplt,*p[0:3]), "r-",alpha=0.5, label="0th order")
# ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "g-", alpha=0.5,label="1st order")
# ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "b-", alpha=0.5, label="2nd order")
# ax.plot(data[:,0]-xmax,data[:,1], "k^", label="Measurement")
# ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Profile Fit")
# ax.set_ylim([0,800])
# xplt1=np.linspace(0,p[1], 2)
# # ax.plot(xplt1, xplt1*0 +distr(p[1],*p[0:2],0), "k-|")
# # ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# # ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
# #ax.text(-0.5, 225, "$\sigma \\approx $"+str("%.2f"%p[1])+" mm", fontsize="large",color="k",backgroundcolor="white")
# plt.legend(loc=0)
# plt.savefig('Overlap.eps', format='pdf',bbox_inches='tight')

f="Overlap.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
ymin = np.amin(data[:,1])
P0=[1000, 2,30, 600, 2, 20, 100, 3,10]
print(data[:,0])
#data[:,0] = (data[:,0]-xmax+10)*2e-3
#data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
print("p=",p)
print("cov=", np.diag(cov)**0.5)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Counts")
# ax.plot(xplt-xmax, gauss(xplt,*p[0:3]), "r-", label="0th order")
# ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "g-", label="1st order")
# ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "b-", label="2nd order")
ax.plot(data[:,0]-xmax,data[:,1], "k-^", label="Measurement")
ax.text(-0.5, 120, "0", fontsize="large",color="k",backgroundcolor="white")
ax.text(12, 120, "+1", fontsize="large",color="k",backgroundcolor="white")
ax.text(21, 120, "+2", fontsize="large",color="k",backgroundcolor="white")
plt.legend(loc=0)
ax.set_ylim([0,800])
plt.savefig('Overlap_no_fit.eps', format='eps',bbox_inches='tight')