#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:12:29 2022

@author: exp-k03
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
from matplotlib.gridspec import GridSpec
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
plt.rcParams['font.size'] = 10
sorted_fold_path="/home/aaa/Desktop/Thesis2/Sorted data/" #insert folder of sorted meausements files
file_name=sorted_fold_path + "Safety plots/5RCWA_eta_TE_GS__0.5__d_322.4181085662225__e1_15.597184423005746__e2_2.339577663450862__e3_0.0__phi2_0.00__phi3_0.00.dat"
diff_eff=np.loadtxt(file_name)
def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)

n_diff= 2 #number of peaks for each side, for example: n=2 for 5 diffracted waves
lam=3.5e-3
LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM
bcr1=8.0 #scattering lenght x density
bcr2=1.2
bcr3=0.
n_0 =1.
phi=0
phi1=0
d0=78
d=322.4181085662225#d0/np.cos((0*rad))
th=diff_eff[:,0]*rad#np.linspace(-0.02,0.02,50)#[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]
S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
eta=S.copy().real
eta_aus=eta.copy()
sum_diff = np.zeros(len(th))
b=2*pi/lam #beta value 
n_1 = bcr1*2*pi/b**2
n_2 = bcr2*2*pi/b**2
n_3 = bcr3*2*pi/b**2
for t in range(len(th)):
    A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
    for i in range(len(A[0])):
        A[i][i]=-dq_j(th[t],i-n_diff,G,b)
        if(i+1<len(A[0])):
            A[i][i+1]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
            A[i+1][i]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
        if(i+2<len(A[0]) and bcr2!=0):
            A[i][i+2]=b**2*n_0*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
            A[i+2][i]=b**2*n_0*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
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
        eta[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
    sum_diff[t] = sum(eta[:,t])
fig, ax = plt.subplots(3,figsize=(10,10))
ax[0].plot(th,eta[n_diff,:],"-k", label="Fit")
ax[0].plot(diff_eff[:,0]*rad,diff_eff[:,n_diff+1],"--w", label="Fit")
#ax[0].set_ylim([np.amin(diff_eff_fit[2,:])-0.4,np.amax(diff_eff_fit[2,:])])
#ax[0].legend(loc=(5))
for i in range(1,3):
    ax[i].plot(th,eta[n_diff-i,:],"-k", label="Fit (-"+str(i)+")")
    ax[i].plot(th,diff_eff[:,n_diff+1+i],"-",color = (0.8,0,0), label="Fit (+"+str(i)+")")  
    ax[i].plot(th,eta[n_diff+i,:],"--w", label="Fit (+"+str(i)+")") 
    ax[i].plot(th,diff_eff[:,n_diff+1-i],"--w", label="Fit (-"+str(i)+")")

