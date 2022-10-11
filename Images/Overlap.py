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
plt.rcParams['font.size'] = 15
def distr(x,A1,sx1,x01,A2,sx2,x02,A3,sx3,x03):
    return A1/(sx1)*np.exp(-0.5*(x-x01)**2/sx1**2)+A2/(sx2)*np.exp(-0.5*(x-x02)**2/sx2**2)+A3/(sx3)*np.exp(-0.5*(x-x03)**2/sx3**2)

def gauss(x,A,sx,x0):
    return A/(sx)*np.exp(-0.5*(x-x0)**2/sx**2)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
f="/home/aaa/Desktop/Thesis2/Images/Overlap.csv"
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

# ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Profile Fit $P(x)$")
# ax.plot(xplt-xmax, gauss(xplt,*p[0:3]), "-",color=(0.7,0.,0.), alpha=0.8, label="$G_0$(x)")
# ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "-", color=(0.,0.7,0.), alpha=0.8,label="$G_{1}(x)$")
# ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "-", color=(0.,0.,0.7), alpha=0.8, label="$G_{2}(x)$")
# ax.plot(xplt-xmax, distr(xplt,*p), "k--")
# ax.plot(data[:,0]-xmax,data[:,1], "k^", label="Data")
# ax.set_ylim([0,800])
# plt.show()
# #xplt1=np.linspace(0,p[1], 2)
# #ax.plot(xplt1, xplt1*0 +distr(p[1],*p[0:2],0), "k-|")
# # ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# # ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
# # ax.text(-0.5, 225, "$\sigma \\approx $"+str("%.2f"%p[1])+" mm", fontsize="large",color="k",backgroundcolor="white")
# plt.legend(loc=0)
# plt.savefig('/home/aaa/Desktop/Thesis2/Images/Overlap.eps', format='pdf',bbox_inches='tight')

# f="Overlap.csv"
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
ax.plot(data[:,0]-xmax,data[:,1], "k-^", label="Data")
ax.text(-0.5, 130, "0", fontsize="large",color="k",backgroundcolor="none")
ax.text(11, 130, "+1", fontsize="large",color="k",backgroundcolor="none")
ax.text(20, 130, "+2", fontsize="large",color="k")#,backgroundcolor="white")
plt.legend(loc=0)
ax.set_ylim([0,800])
plt.savefig('Overlap_no_fit.eps', format='eps',bbox_inches='tight')