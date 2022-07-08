#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:22:07 2022

@author: aaa
"""

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
import matplotlib.pyplot as plt
import matplotlib as mpl
import socket
import shutil
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare as cs
import scipy.integrate as integrate
import math
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from datetime import datetime
from multiprocessing import Pool
pi=np.pi
rad=pi/180

sorted_fold_path="/home/aaa/Desktop/thesis_L1/Sorted data/" #insert folder of sorted meausements files
allmeasurements = sorted_fold_path+"All measurements/"
allrenamed = allmeasurements +"All renamed/"
allmatrixes = allmeasurements + "All matrixes/"
allpictures = allmeasurements + "All pictures/"
allrawpictures = allmeasurements + "All raw pictures/"
alldata_analysis = allmeasurements + "All Data Analysis/"
allcropped_pictures = alldata_analysis + "All Cropped Pictures/"
allcontrolplots = alldata_analysis + "All Control plots/"
allcontrolfits = alldata_analysis + "All Control Fits/"
tiltangles=[0,40,48,61,69,71,79,80,81]
foldername=[]
for i in range(len(tiltangles)):
    foldername.append(str(tiltangles[i])+"deg")
foldername.append("79-4U_77c88deg")
foldername.append("79-8U_76c76deg")
foldername.append("79-12U_75c64deg")
foldername.append("79-16U_74c52deg")
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measurement
"""
This block fits the diffraction efficiencies n(x)= n_0 + n_1 cos(Gx)
"""
##############################################################################
"""
Wavelenght distribution: Exponentially Modified Gaussian
"""
def func(l,A,mu,sig):
    return A/(2.)*np.exp(A/(2.)*(2.*mu+A*sig**2-2*l))
def rho(l,A,mu,sig):
    return func(l,A,mu,sig)*erfc((mu+A*sig**2-l)/(np.sqrt(2)*sig))

# def rho(l,A,mu,sigma):
#     sigma=sigma+l*0.1
#     mu=mu+1/lambda_par
#     return 1/((2*pi)**0.5*sigma)*np.exp(-(l-mu)**2/(2*sigma**2))
tau=0.0007	#+/-	147.471394720765
mu=2.2e-3#0.004632543663155012	#+/-	5.46776175965519e-05
sigma=0.0003
M = mu+tau
sigma1=(sigma**2+tau**2)**0.5
lambda_par=1/tau
##############################################################################

"""
Angular distribution: Gaussian
"""
div=0.0015/3
def ang_gauss(x):
    sig=div
    return 1/((2*pi)**0.5*sig)*np.exp(-x**2/(2*sig**2))
gx=np.arange(-3*div,3*div, 1e-6)
gauss_conv = ang_gauss(gx)/sum(ang_gauss(gx))

##############################################################################

n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves

LAM= 0.5 #grating constant in micrometers
G0=2*pi/LAM
print(G0)
bcr1=7.5#scattering lenght x density
bcr2=0.7
bcr3=0
n_0 =1.
phi=0
phi1=0
d0=78
k=10
x00=-0.000/rad
def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)

wlp=5e-9
print(foldername[k])
data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_1line.mpa',skiprows=1)
fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_1line.mpa',skiprows=1)
p=fit_res[0]
print(p)
for i in range(len(diff_eff[:,0])): 
    s=sum(diff_eff[i,2::2])
    diff_eff[i,2:]=diff_eff[i,2:]/s
# diff_eff_err= diff_eff[:,3::2]
# diff_eff_err=np.divide(diff_eff_err,diff_eff[:,2::2])
# diff_eff_err[np.isnan(diff_eff_err)]=0
diff_eff_fit=np.zeros((5, len(diff_eff[:,5])))
diff_eff_fit[2,:]=diff_eff[:,2*2+2].copy()
for i in range(1,3):
    diff_eff_fit[2-i,:]=diff_eff[:,6-2*i].copy()
    diff_eff_fit[2+i,:]=diff_eff[:,6+2*i].copy()
wl=np.linspace(mu-2.5*sigma,mu+1/lambda_par+3.5*sigma1, 10000)
a = rho(wl,lambda_par, mu, sigma)/sum(rho(wl,lambda_par, mu, sigma))
spl = UnivariateSpline(wl, a, k=3, s=0)
d=spl.antiderivative()(wl)
y=np.arange(d[d==np.amin(d)],d[d==np.amax(d)]+wlp,  wlp)
# print("points=",len(y))
x=np.zeros(len(y))
for i in range(len(y)):
    aus =abs(spl.antiderivative()(wl)-y[i])
    x[i]=wl[aus==np.amin(aus)]
fig = plt.figure(figsize=(10,10))
ax= fig.add_subplot()
ax.set_title(foldername[k])
ax.plot(wl,d/np.amax(d))
ax.plot(wl,a/np.amax(a))
ax.plot(x,x*0,"k.")
ax.set_xlim([0,0.011])
a=rho(x,lambda_par, mu, sigma)/sum(rho(x,lambda_par, mu, sigma))
ax.plot(x,a/np.amax(a),"g.")
  
def plot_func(x):
    d=d0/np.cos((tilt[k])*rad)
    wl=np.linspace(mu-2.5*sigma, mu+1/lambda_par+3.5*sigma1, 10000)
    a = rho(wl,lambda_par, mu, sigma)/sum(rho(wl,lambda_par, mu, sigma))
    spl = UnivariateSpline(wl, a, k=4, s=0)
    I=spl.antiderivative()(wl)
    y=np.arange(I[I==np.amin(I)],I[I==np.amax(I)]+wlp,  wlp)
    xp=np.zeros(len(y))
    for i in range(len(y)):
        aus =abs(spl.antiderivative()(wl)-y[i])
        xp[i]=wl[aus==np.amin(aus)]
    wl=xp.copy()
    a=rho(xp,lambda_par, mu, sigma)/sum(rho(xp,lambda_par, mu, sigma))
    th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]
    S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
    eta=S.copy().real
    eta_aus=eta.copy()
    sum_diff = np.zeros(len(th))
    for l in range(len(wl)):
        lam=wl[l] #single wavelenght in micrometers
        b=2*pi/lam #beta value 
        n_1 = bcr1*2*pi/b**2
        n_2 = bcr2*2*pi/b**2
        n_3 = bcr3*2*pi/b**2
        for t in range(len(th)):
            G=G0#*(np.cos((th[t]-th[0])))    
            A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
            for i in range(len(A[0])):
                A[i][i]=-dq_j(th[t],i-n_diff,G,b)
                if(i+1<len(A[0])):
                    A[i][i+1]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+1][i]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
                if(i+2<len(A[0]) and bcr2!=0):
                    A[i][i+2]=-b**2*n_0*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+2][i]=-b**2*n_0*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
                if(i+3<len(A[0]) and bcr3!=0):
                    A[i][i+3]=b**2*n_0*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+3][i]=b**2*n_0*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
            A=-1j*A
            w,v = np.linalg.eig(A)
            v0=np.zeros(2*n_diff+1)
            v0[n_diff]=1
            c = np.linalg.solve(v,v0)
            for i in range(len(w)):
                v[:,i]=v[:,i]*c[i]*np.exp(w[i]*d)
            for i in range(len(S[:,0])):
                S[i,t] = sum(v[i,:])
        for t in range(len(th)):
            for i in range(2*n_diff+1):
                eta_aus[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
            sum_diff[t] = sum(eta[:,t])
        eta+=eta_aus*a[l]
    eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
    x_int=np.arange(th[0],th[-1], 1e-6)
    for i in range(n_diff*2+1):
        f_int = interp1d(th,eta[i,:], kind="cubic")
        conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
        f_int = interp1d(x_int,conv, kind="cubic")
        eta_ang[i,:]=f_int(x*rad)
    print(G)
    return eta_ang
# print(wlpoints)
thx=diff_eff[:,0]*rad
eta=plot_func(diff_eff[:,0])
fig, ax = plt.subplots(n_diff+2,figsize=(10,10))
ax[0].set_title(foldername[k])
ax[0].plot(diff_eff[:,0]*rad,diff_eff_fit[2,:], 'ro')
# ax[0].errorbar(diff_eff[:,0]*rad,diff_eff_fit[2,:], yerr=diff_eff[:,7])
ax[0].plot(thx,eta[n_diff,:],"1-")
for i in range(1,n_diff+1):
    if i<3:
        ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6-2*i],'o')
        # ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6-2*i], yerr=diff_eff[:,7-2*i]*10)
        ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6+2*i],'o')
        # ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6+2*i], yerr=diff_eff[:,7+2*i]*10)
    ax[i].plot(thx,eta[n_diff-i,:],"1-")
    ax[i].plot(thx,eta[n_diff+i,:],"1-")   
# ax[n_diff+1].plot(th, sum_diff)
# ax[n_diff+1].set_ylim([0.5,1.5])
#   plt.errorbar(diff_eff[:,0],diff_eff[:,2*j+2],yerr=diff_eff[:,2*j+1],capsize=1)
