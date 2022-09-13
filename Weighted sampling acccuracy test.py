
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:28:47 2022

@author: aaa
"""
from scipy.special import erfc
from scipy.stats import exponnorm
from scipy.stats import norm
from scipy.optimize import curve_fit as fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import inspect, os, time
from scipy.interpolate import interp1d

pi=np.pi
rad=pi/180
n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves
LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM #grating vector
bcr1=5.0 #scattering lenght x density
bcr2=0. 
bcr3=0.
n_0 =1.
phi=0 #phase shift bcr2
phi1=0 #phase shift bcr3
d0=78 #sample thickness

def rho(x, sk,x0, sig):
    g = exponnorm(loc=x0, K=sk, scale=sig)
    return g.pdf(x)
div=0.00035
def ang_gauss(x):
    g=norm(loc=0,scale=div)
    return g.pdf(x)
gx = np.arange(norm.ppf(0.001, loc=0, scale=div), norm.ppf(0.999, loc=0, scale=div), 1e-6)
gauss_conv = ang_gauss(gx)/sum(ang_gauss(gx))

def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)
def plot_func(x, bcr1, bcr2, bcr3, mu1, sigma, tau, x00,zeta0, phi, phi1, wlp):
    th=[x[0]-3*div,*x,x[-1]+3*div]
    phi=phi*pi
    phi1=phi1*pi
    d=d0/np.cos(zeta0*rad)
    wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
    a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
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
                eta_aus[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
            sum_diff[t] = sum(eta[:,t])
        eta+=eta_aus*a[l]
    eta_ang = np.zeros((2*n_diff+1,len(x)))
    x_int=np.arange(th[0],th[-1], 1e-6)
    for i in range(n_diff*2+1):
        f_int = interp1d(th,eta[i,:], kind="cubic")
        conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
        f_int = interp1d(x_int,conv, kind="cubic")
        eta_ang[i,:]=f_int(x)
    return eta_ang
x=np.linspace(-0.02,0.02, 100)
Delta = plot_func(x, 8, 1, 0, 3.5e-3, 1e-3, 10, 0, 80, 1*pi, 0, 1e-2)-plot_func(x, 8, 1, 0, 3.5e-3, 1e-3, 10, 0, 80, 1*pi, 0, 1e-3)
wl=exponnorm.ppf(np.arange(0.01,0.99,3e-5), K=10, loc=3.5e-3, scale=1e-3)
Dx = wl[1:]-wl[:-1]
print("Dx avg=",np.average(Dx),"; Dx min=", np.amin(Dx),"; Dx max=", np.amax(Dx))
print(np.average(abs(Delta)),np.amin(abs(Delta)),np.amax(abs(Delta)))