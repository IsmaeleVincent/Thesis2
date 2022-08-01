#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:05:32 2022

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:35:07 2022

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
from scipy.stats import chisquare
pi=np.pi
rad=pi/180

orig_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/"
fold_name = "D0043" # "NP829"
sorted_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/Sorted data/" #insert folder of sorted meausements files
renamed = sorted_fold_path+fold_name+"/Renamed/"
matrixes = sorted_fold_path+fold_name+"/Matrixes/"
pictures = sorted_fold_path+fold_name+"/Pictures/"
rawpictures = sorted_fold_path+fold_name+"/Raw pictures/"
th_matrixes = sorted_fold_path+fold_name+"/Theta matrixes/"
th_pictures = sorted_fold_path+fold_name+"/Theta pictures/"
th_rawpictures = sorted_fold_path+fold_name+"/Theta raw pictures/"
data_analysis = sorted_fold_path+fold_name+"/Data analysis/"
n_meas=175#150 #number of measurements files for each folder
n_pixel = 16384 #number of pixels in one measurement
foldername=[fold_name]

n_diff= 3#number of peaks for each side, for example: n=2 for 5 diffracted waves

LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM
bcr1=15.196463433975495#scattering lenght x density
bcr2=-6.581479166366684
bcr3=2.8539888153438504
phi=0
phi1=0
d=34.62821383123897
mu=2e-3
L=1e8
def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)
thx=np.linspace(-2*rad,2*rad,100)
def plot_func(th, bcr1, bcr2, bcr3, lam,d,L):
    d=d/np.cos(45*rad)
    L=L/np.cos(45*rad)
    S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
    eta=S.copy().real
    b=2*pi/lam #beta value 
    n_1 = bcr1*2*pi/b**2
    n_2 = bcr2*2*pi/b**2
    n_3 = bcr3*2*pi/b**2
    z_step=np.linspace(0,d,10)
    v0=np.zeros(2*n_diff+1)
    v0[n_diff]=1
    v0_th=S.copy()
    for i in range(len(th)):
        v0_th[:,i]=v0
    for z in z_step:
        # n_1 = n_10*np.exp(-z/L)
        # n_2 = n_20*np.exp(-z/L)
        # n_3 = n_30*np.exp(-z/L)
        for t in range(len(th)):
            A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
            for i in range(len(A[0])):
                A[i][i]=-dq_j(th[t],i-n_diff,G,b)
                if(i+1<len(A[0])):
                    A[i][i+1]=b**2*n_1/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+1][i]=b**2*n_1/(2*k_jz(th[t],i-n_diff,G,b))
                if(i+2<len(A[0]) and bcr2!=0):
                    A[i][i+2]=b**2*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+2][i]=b**2*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
                if(i+3<len(A[0]) and bcr3!=0):
                    A[i][i+3]=b**2*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
                    A[i+3][i]=b**2*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
            A=-1j*A
            w,v = np.linalg.eig(A)
            c = np.linalg.solve(v,v0_th[:,t])
            for i in range(len(w)):
                v[:,i]=v[:,i]*c[i]*np.exp(w[i]*z)
            for i in range(len(S[:,0])):
                S[i,t] = sum(v[i,:])*np.exp(-1j*dq_j(th[t],i-n_diff,G,b)*z)
        v0_th=S.copy()
    for t in range(len(th)):
        for i in range(2*n_diff+1):
            eta[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
        # print(np.amin(eta[n_diff,:]))
    return eta
p=[bcr1,bcr2,bcr3, mu,d,L]
eta=plot_func(thx,*p)
fig, ax = plt.subplots(n_diff+1,figsize=(10,12))
ax[0].set_title(fold_name)
ax[0].plot(thx,eta[n_diff,:],"1--k", label="Fit")
#ax[0].plot(thx,eta1[n_diff,:],"1--b", label="Fit")
#ax[0].legend(loc=(5))
for i in range(1,n_diff+1):
    ax[i].plot(thx,eta[n_diff-i,:],"1--k", label="Fit (-"+str(i)+")")
    ax[i].plot(thx,eta[n_diff+i,:],"1--",color = (0.8,0,0), label="Fit (+"+str(i)+")")
    # ax[i].plot(thx,eta1[n_diff-i,:],"1--b", label="Fit (-"+str(i)+")")
    # ax[i].plot(thx,eta1[n_diff+i,:],"1--",color = (0,0.8,0), label="Fit (+"+str(i)+")")
    # ax[i].legend()
p_name=["$(b_c \\rho)_1$","$(b_c \\rho)_2$", "$(b_c \\rho)_3$", "$\lambda$", "d", "L"]
p_units=[" $1/\mu m^2$"," $1/\mu m^2$"," $1/\mu m^2$"," nm"," $\mu m$"," $\mu m$"]
text = "Parameters"
p[3]*=1e3
for i in range(len(p)):
    text+= "\n" + p_name[i] + "=" + str("%.3f" % (p[i],))+ p_units[i]
ax[0].text(thx[0],np.amin(eta[n_diff,:]), text,  bbox=dict(boxstyle="square", ec=(0, 0, 0), fc=(1,1,1)))
