#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:12:29 2022

@author: exp-k03
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
Wavelenght distribution: Exponentially Modified Gaussian
"""
def func(l,A,mu,sig):
    return A/(2.)*np.exp(A/(2.)*(2.*mu+A*sig**2-2*l))
def rho(l,A,mu,sig):
    return func(l,A,mu,sig)*erfc((mu+A*sig**2-l)/(np.sqrt(2)*sig))

tau=0.00144	#+/-	147.471394720765
mu=2.5e-3#0.004632543663155012	#+/-	5.46776175965519e-05
sigma=0.0007
##############################################################################

"""
Angular distribution: Gaussian
"""
div=0.00032
def ang_gauss(x):
    sig=div
    return 1/((2*pi)**0.5*sig)*np.exp(-x**2/(2*sig**2))
gx=np.arange(-3*div,3*div, 1e-6)
gauss_conv = ang_gauss(gx)/sum(ang_gauss(gx))


def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)

n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves
LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM
bcr1=8.0 #scattering lenght x density
bcr2=1.2
bcr3=0.
n_0 =1.
phi=0
phi1=0
d0=78

tilt=np.linspace(0,81,50)#np.sort(tilt)
pendol = np.zeros((len(tilt),3))
pendol[:,0] = tilt
k=-1
wlp=5e-8
for zeta in tilt:
    k+=1
    lambda_par=1/tau
    sigma1=(sigma**2+tau**2)**0.5
    d=d0/np.cos((zeta)*rad)
    # d=d/np.cos((tilt[k])*rad)
    wl=np.linspace(mu-2.5*sigma, mu+1/lambda_par+3.5*sigma, 10000)
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
    th=np.linspace(-0.02,0.,100)
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
    eta_ang = np.zeros((2*n_diff+1,len(th)))
    x_int=np.linspace(th[0],th[-1], 10000)
    for i in range(n_diff*2+1):
        f_int = interp1d(th,eta[i,:], kind="cubic")
        conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
        f_int = interp1d(x_int,conv, kind="cubic")
        eta_ang[i,:]=f_int(th)
    # fig, ax = plt.subplots(3,figsize=(10,10))
    # ax[0].plot(th,eta_ang[n_diff,:],"-k", label="Fit")
    # for i in range(1,3):
    #     ax[i].plot(th,eta_ang[n_diff-i,:],"-k", label="Fit (-"+str(i)+")")  
    #     ax[i].plot(th,eta_ang[n_diff+i,:],"--w", label="Fit (+"+str(i)+")") 

    pendol[k,1]=np.amax(eta_ang[n_diff-1,:])
    pendol[k,2]=np.amax(eta_ang[n_diff-2,:])

plt.plot(pendol[:,0],pendol[:,1],"--k")
plt.plot(pendol[:,0],pendol[:,2],"--", color=(0.8,0,0))

krange=range(len(foldername))#[0,2,3,4,5]#range(7,len(foldername))#[0,2,3,4,5] #np.arange(len(foldername))#
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
pendol = np.zeros((len(foldername),5))
pendol[:,0]= tilt.copy()
for k in krange:
    # print(foldername[k])
    nowf=datetime.now()
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
    diff_eff[:,3::2]=diff_eff[:,2::2]**0.5
    diff_eff_aus=diff_eff[:,2::2].copy()
    diff_eff_aus_err=diff_eff[:,3::2].copy()
    diff_eff_aus[diff_eff_aus==0]=1
    for i in range(len(diff_eff[:,0])):
        s=sum(diff_eff[i,2::2])
        diff_eff[i,2:]=diff_eff[i,2:]/s
    diff_eff_fit=diff_eff[:,2::2].copy()
    diff_eff_err=(diff_eff_fit**2+diff_eff_fit)
    for i in range(len(diff_eff_err[:,0])):
        s=sum(diff_eff_aus_err[i,:])
        for j in range(len(diff_eff_err[0,:])):
            diff_eff_err[i,j]=diff_eff_err[i,j]*s/diff_eff_aus[i,j]
    diff_eff_err[diff_eff_err==0]=0.01
    diff_eff[:,3::2]=diff_eff_err
    pendol[k,1]=np.amax(diff_eff_fit[:,1])
    pendol[k,2]=diff_eff_err[:,1][diff_eff_fit[:,1]==pendol[k,1]]
    pendol[k,3]=np.amax(diff_eff_fit[:,0])
    pendol[k,4]=diff_eff_err[:,0][diff_eff_fit[:,0]==pendol[k,3]]

pendol[:,0]=pendol[np.argsort(pendol[:,0],axis=0),0]
pendol[:,1]=pendol[np.argsort(pendol[:,0],axis=0),1]
plt.errorbar(pendol[:,0],pendol[:,1],pendol[:,2], fmt="^k")
plt.errorbar(pendol[:,0],pendol[:,3],pendol[:,4], fmt="^", color=(0.8,0,0))

