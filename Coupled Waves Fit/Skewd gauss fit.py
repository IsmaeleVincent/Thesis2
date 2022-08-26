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
from scipy.special import erfc

def rho(x,B,A,x0, sk, sig):
    g=skewnorm(loc=x0,a=sk,scale=sig)
    return B+A*g.pdf(x)
def func(l,A,mu,sig):
    return A/(2.)*np.exp(A/(2.)*(2.*mu+A*sig**2-2*l))
def rho1(l,B,A1,A,mu,sig):
    return B+A1*func(l,A,mu,sig)*erfc((mu+A*sig**2-l)/(np.sqrt(2)*sig))

# file_name="/home/aaa/Desktop/Thesis2/Wavelength distribution/measuredVCNSpectr.dat"
# wl_dist=np.loadtxt(file_name) 
# x=wl_dist[:,0]
# y=wl_dist[:,1]
# P0 = [30,350,3.5e-9,0, 1e-9]
# p,cov=fit(rho,x,y, p0=P0)
# # P01 = [30,350,1e9/0.1,3.6e-9, 0.3e-9]
# # p1,cov1=fit(rho1,x,y, p0=P01)
# plt.plot(x,y,".")
# plt.plot(x,rho(x,*p), label="Skewed gaussian")
# #plt.plot(x,rho1(x,*p1), label="EMG")
# print(p)
# plt.vlines(p[2],0,350)
# plt.legend()

p = [0,1,2.3e-3,2,1e-3]
if p[3]<1:
    x0=p[2]-3.5*p[4]
else:
    x0=p[2]-3*p[4]/p[3]
x1=p[2]+3.5*p[4]
x=np.linspace(x0,x1,1000)
plt.plot(x,rho(x,*p), label="Skewed gaussian")
#plt.plot(x,rho1(x,*p1), label="EMG")
print(p)
plt.vlines(p[2],0,rho(p[2],*p))
plt.vlines(x0,0,50)#rho(x0,*p))
plt.vlines(x1,0,50)#rho(x1,*p))
print(rho(x0,*p)/np.amax(rho(x,*p)),rho(x1,*p)/np.amax(rho(x,*p)))
plt.legend()

