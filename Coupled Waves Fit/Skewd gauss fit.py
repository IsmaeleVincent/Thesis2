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
from scipy.interpolate import UnivariateSpline


def func(l,A,mu,sig):
    return A/(2.)*np.exp(A/(2.)*(2.*mu+A*sig**2-2*l))
def rho(l,B,C,mu,A,sig):
    return B+C*func(l,A,mu,sig)*erfc((mu+A*sig**2-l)/(np.sqrt(2)*sig))

# def rho(x,B,A,x0, sk, sig):
#     g=skewnorm(loc=x0,a=sk,scale=sig)
#     return B+A*g.pdf(x)
def rho1(x,B,A,x0, sk, sig):
    g=exponnorm(loc=x0,K=sk,scale=sig)
    return B+A*g.pdf(x)
file_name="/home/aaa/Desktop/Thesis2/Wavelength distribution/measuredVCNSpectr.dat"
wl_dist=np.loadtxt(file_name) 
x=wl_dist[:,0]*1e6
y=wl_dist[:,1]
P0 = [30,350,3.6e-3,1e3, 1e-4]
p,cov=fit(rho,x,y, p0=P0)
P01 = [30,1e-3,3.5e-3,10, 2e-4]
B=([0,0,2e-3,1,1e-4],[100,1,4e-3,20,1e-3])
p1,cov1=fit(rho1,x,y, p0=P01, bounds=B)
plt.plot(x,y,".")
# plt.plot(x,rho(x,*p), label="Skewed gaussian")
plt.plot(x,rho1(x,*p1), label="EMG")
spline = UnivariateSpline(x, rho1(x,*p1)-np.amax(rho1(x,*p1))/2, s=0)
r1, r2 = spline.roots()
fwhm=abs(r2-r1)
plt.vlines([r1,r2],0,300)
plt.vlines(exponnorm.ppf(0.5,loc=p1[2],K=p1[3],scale=p1[4]),0,300)
print(fwhm)
print("dl/l=",fwhm/exponnorm.ppf(0.5,loc=p1[2],K=p1[3],scale=p1[4]))
# pp=[8.43918354e+00, 4.65054398e-01, 1.0e-04, 3.6e-03, 1e-05]
# x1=np.linspace(0,10e-3, 1000)
# plt.plot(x1,rho1(x1,*pp), label="EMG")

print(p1)
print(1/p[3]*1e3)
print(1/(p[3]*p[-1]))
lmax=x[rho1(x,*p1)==np.amax(rho1(x,*p1))]
print(lmax)
plt.vlines(lmax,0,np.amax(rho1(x,*p1)), label="max 1="+str("%.2e"%(lmax,)))
plt.legend()

# p = [0,1,2.3e-3,2,1e-3]
# if p[3]<1:
#     x0=p[2]-3.5*p[4]
# else:
#     x0=p[2]-3*p[4]/p[3]
# x1=p[2]+3.5*p[4]
# x=np.linspace(x0,x1,1000)
# plt.plot(x,rho(x,*p), label="Skewed gaussian")
# #plt.plot(x,rho1(x,*p1), label="EMG")
# print(p)
# plt.vlines(p[2],0,rho(p[2],*p))
# plt.vlines(x0,0,50)#rho(x0,*p))
# plt.vlines(x1,0,50)#rho(x1,*p))
# print(rho(x0,*p)/np.amax(rho(x,*p)),rho(x1,*p)/np.amax(rho(x,*p)))
# plt.legend()

