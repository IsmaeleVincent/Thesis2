"""
This module defines the vector field for 5 coupled wave equations
(without a decay, Uchelnik) and first and second harmonics in the modulation; phase: 0 or pi (sign of n2).
Fit parameters are: n1,n2, d, and wavelength; 
Fit 5/(5) orders!
!!!Data: X,order,INTENSITIES
Fit  background for second orders , first and subtract it for zero orders (background fixed)
"""
from scipy.integrate import ode
from scipy import integrate
import numpy as np
from numpy.linalg import eig,solve
import inspect,os,time
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.special import erfc
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import socket
import shutil
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare as cs
import scipy.integrate as integrate
import math
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

pi=np.pi
rad=pi/180


"""
Angular distribution: Gaussian
"""
sig=0.0006/2
def rho(x,x0):
    g=norm(loc=x0)
    return g.pdf(x)


##############################################################################
mu=4
wl=np.linspace(mu-3.5,mu+3.5, 100000)
a = rho(wl,mu)
spl = UnivariateSpline(wl, a, k=3, s=0)
d=spl.antiderivative()(wl)
s=20
y=np.linspace(d[d==np.amin(d)],d[d==np.amax(d)], s)
x=np.zeros(len(y))
guess=0
print(len(y))
for i in range(len(y)):
    aus =abs(spl.antiderivative()(wl)-y[i])
    x[i]=wl[aus==np.amin(aus)]
fig, ax = plt.subplots(1,2,figsize=(10,2), sharey=True,sharex=False)
#ax[0].plot(wl,d)
ax[0].plot(wl,rho(wl,mu),"k")
ax[0].plot(x,rho(x,mu),"^k")#, color = (0.8,0,0))
ax[0].plot(x,rho(x,mu)*0,".k")
ax[0].vlines(x, 0, rho(x,mu), color= (0.5,0.5,0.5))#, ls="dashed") 
ax[1].plot(wl,rho(wl,mu),"k")
y=np.linspace(x[0],x[-1], s)
ax[1].plot(y,rho(y,mu),"^k")#, color = (0.8,0,0))
ax[1].plot(y,rho(y,mu)*0,".k")
ax[1].vlines(y, 0, rho(y,mu),color= (0.5,0.5,0.5))#, ls="dashed")
#ax[0].plot(x,rho(x,mu),".", color = (0.8,0,0))
ax[0].set_ylim([0,np.amax(a)+0.01])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('Weighted_sampling.eps', format='eps',bbox_inches='tight')

# a = rho1(wl,lambda_par, mu, sigma)/sum(rho1(wl,lambda_par, mu, sigma))
# from scipy.interpolate import UnivariateSpline
# spl = UnivariateSpline(wl, a, k=3, s=0)
# d=spl.antiderivative()(wl)
# s=100
# y=np.linspace(d[d==np.amin(d)],d[d==np.amax(d)],  s)
# x=np.zeros(s)
# for i in range(s):
#     aus =abs(spl.antiderivative()(wl)-y[i])
#     x[i]=wl[aus==np.amin(aus)]
# plt.plot(wl,d/np.amax(d))
# plt.plot(wl,a/np.amax(a))
# plt.plot(x,x*0,"k.")
# a=rho1(x,lambda_par, mu, sigma)/sum(rho1(x,lambda_par, mu, sigma))
# plt.plot(x,a/np.amax(a),"g.")

# print(a[0],a[-1])
# data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
# diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff.mpa',skiprows=1)
# x1=diff_eff[:,0]*rad
# th=np.linspace(x1[0]-3*div,x1[-1]+3*div,10000)
# print(x1)
# asd=np.zeros(10000)
# for i in range(len(x1)):
#     asd += ang_gauss(th,x1[i])
# spl=UnivariateSpline(th, asd, k=3, s=0)
# d=spl.antiderivative()(th)
# plt.plot(th, asd/np.amax(asd))
# plt.plot(th, d/np.amax(d))
# s=len(x1)*100
# y=np.linspace(d[d==np.amin(d)],d[d==np.amax(d)],  s)
# x=np.zeros(s)
# for i in range(s):
#     aus =abs(spl.antiderivative()(th)-y[i])
#     x[i]=th[aus==np.amin(aus)]
# plt.plot(x,x*0,"k.")
# plt.plot(x1,x1*0,"r.")


# plt.plot(qwe) 
# print(sum(qwe)/len(qwe))