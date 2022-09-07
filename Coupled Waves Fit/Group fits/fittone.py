#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:05:24 2022

@author: aaa
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
from scipy.stats import norm
from scipy.stats import cosine
from scipy.stats import exponnorm

pi=np.pi
rad=pi/180
plt.rcParams['font.size'] = 10
sorted_fold_path="/home/ismaele/Desktop/Thesis2/Sorted data/" #insert folder of sorted meausements files
allmeasurements = sorted_fold_path+"All measurements/"
allrenamed = allmeasurements +"All renamed/"
allmatrixes = allmeasurements + "All matrixes/"
allpictures = allmeasurements + "All pictures/"
allrawpictures = allmeasurements + "All raw pictures/"
alldata_analysis = allmeasurements + "All Data Analysis/"
allcropped_pictures = alldata_analysis + "All Cropped Pictures/"
allcontrolplots = alldata_analysis + "All Control plots/"
allcontrolfits = alldata_analysis + "All Control Fits/"
allfits_plots= [alldata_analysis + "All fits plots/Juergen/bcr_1_2_3/",
                alldata_analysis + "All fits plots/Martin/bcr_1_2_3/",
                alldata_analysis + "All fits plots/Christian/bcr_1_2_3/",
                alldata_analysis + "All fits plots/All groups/bcr_1_2_3/"]
alldiff_eff_fit=allfits_plots.copy()
allwl_plots=allfits_plots.copy()

for g in range(4):
    if os.path.exists(allfits_plots[g]):
            shutil.rmtree(allfits_plots[g])
    os.makedirs(allfits_plots[g])
    alldiff_eff_fit[g]+="Diff. effs + fits/"
    if os.path.exists(alldiff_eff_fit[g]):
            shutil.rmtree(alldiff_eff_fit[g])
    os.makedirs(alldiff_eff_fit[g])
    allwl_plots[g]+="WL distributions/"
    if os.path.exists(allwl_plots[g]):
            shutil.rmtree(allwl_plots[g])
    os.makedirs(allwl_plots[g])

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
def rho(l,tau,mu,sig):
    emg=exponnorm(loc=mu,K=tau, scale=sig)
    return emg.pdf(l)
##############################################################################

"""
Angular distribution: Gaussian
"""
div=0.00035
def ang_gauss(x):
    g=norm(loc=0,scale=div)
    return g.pdf(x)
gx = np.arange(norm.ppf(0.001, loc=0, scale=div), norm.ppf(0.999, loc=0, scale=div), 1e-6)
gauss_conv = ang_gauss(gx)/sum(ang_gauss(gx))
# plt.plot(gx,ang_gauss(gx))

##############################################################################

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

measur_groups=[[0,2,3,4,5],[6,7,8,9,10,11,12],[1], range(13)]


for group in [2]: #0 for Juergen, 1 for Martin, 2 for Christian, 3 for all 
    krange=measur_groups[group]
    
    def k_jz(theta, j, G,b):
        k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
        return k_jz
    def dq_j (theta, j, G,b):
        return b*np.cos(theta) - k_jz(theta, j, G, b)
    
    fitting=1
    plotting=1
    extended_plot=1
    save_fit_res=1
    standard=0
    no_div=0
    fit_phi=1

    wlp=5e-3
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_new_diff.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, bcr3, mu1, sigma, tau, x00, zeta0):
            x=diff_eff[:,0]+x00
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
            x_int=np.arange(th[0],th[-1], 1e-6)
            for i in range(n_diff*2+1):
                f_int = interp1d(th,eta[i,:], kind="cubic")
                conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
                f_int = interp1d(x_int,conv, kind="cubic")
                eta_ang[i,:]=f_int(x*rad)
            aaa=eta_ang[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(8) # [*fit_res[0],0 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=1.
            P0[3]=3.5e-3
            P0[4]=0.0002
            P0[5]=5
            P0[6]=0
            P0[7]=0
            
            Bi_groups=[[5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2]]
            Bf_groups=[[10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2]]
            
            B=(Bi_groups[group],Bf_groups[group])
            
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2_3.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")
    
    if (fitting and standard):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)
    
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        # fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_bcr_1_2.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, mu1, sigma, tau, x00, zeta0):
            x=diff_eff[:,0]+x00
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
            x_int=np.arange(th[0],th[-1], 1e-6)
            for i in range(n_diff*2+1):
                f_int = interp1d(th,eta[i,:], kind="cubic")
                conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
                f_int = interp1d(x_int,conv, kind="cubic")
                eta_ang[i,:]=f_int(x*rad)
            aaa=eta_ang[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(7) # [*fit_res[0],0  # fit_res[0] # [*fit_res[0,:-1],0,0]  # fit_res[0] #  [8, 2,0, 2.01e-3, pi,0, 75, 1000, 0.0004] #    [5,0,2.6e-3] # 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=3.5e-3
            P0[3]=0.0002
            P0[4]=5
            P0[5]=0
            P0[6]=0
            Bi_groups=[[5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2]]
            Bf_groups=[[10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2]]
            B=(Bi_groups[group],Bf_groups[group])
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")
    
    if (fitting and standard):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)
        
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        # fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_1_2_3_no_div.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, bcr3, mu1, sigma, tau, x00, zeta0):
            x=diff_eff[:,0]+x00
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=x*rad#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            aaa=eta[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(8) # [*fit_res[0],0  # fit_res[0] # [*fit_res[0,:-1],0,0]  # fit_res[0] #  [8, 2,0, 2.01e-3, pi,0, 75, 1000, 0.0004] #    [5,0,2.6e-3] # 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=1.
            P0[3]=3.5e-3
            P0[4]=0.0002
            P0[5]=5
            P0[6]=0
            P0[7]=0
            
            Bi_groups=[[5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2]]
            Bf_groups=[[10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2]]
            
            B=(Bi_groups[group],Bf_groups[group])
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2_3_no_div.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")
    
    if (fitting and no_div):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)
        
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        # fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_bcr_1_2.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, mu1, sigma, tau, x00, zeta0):
            x=diff_eff[:,0]+x00
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=x*rad
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            # eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
            # x_int=np.arange(th[0],th[-1], 1e-6)
            # for i in range(n_diff*2+1):
            #     f_int = interp1d(th,eta[i,:], kind="cubic")
            #     conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
            #     f_int = interp1d(x_int,conv, kind="cubic")
            #     eta_ang[i,:]=f_int(x*rad)
            aaa=eta[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(7) # [*fit_res[0],0  # fit_res[0] # [*fit_res[0,:-1],0,0]  # fit_res[0] #  [8, 2,0, 2.01e-3, pi,0, 75, 1000, 0.0004] #    [5,0,2.6e-3] # 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=3.5e-3
            P0[3]=0.0002
            P0[4]=5
            P0[5]=0
            P0[6]=0
            Bi_groups=[[5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2]]
            Bf_groups=[[10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2]]
            B=(Bi_groups[group],Bf_groups[group])
            
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")
    
    if (fitting and no_div):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)
    
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_new_diff.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, bcr3, mu1, sigma, tau, x00, zeta0, phi, phi1):
            x=diff_eff[:,0]+x00
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            phi=phi*pi
            phi1=phi1*pi
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
            x_int=np.arange(th[0],th[-1], 1e-6)
            for i in range(n_diff*2+1):
                f_int = interp1d(th,eta[i,:], kind="cubic")
                conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
                f_int = interp1d(x_int,conv, kind="cubic")
                eta_ang[i,:]=f_int(x*rad)
            aaa=eta_ang[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(10) # [*fit_res[0],0 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=1.
            P0[3]=3.5e-3
            P0[4]=0.0002
            P0[5]=5
            P0[6]=0
            P0[7]=0
            P0[8]=1
            P0[9]=0
            Bi_groups=[[5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0, 0],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0, 0],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0, 0],
                      [5, 0, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0, 0]]
            Bf_groups=[[10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2,2,2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2,2,2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2,2,2],
                      [10, 5, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2,2,2]]
            
            B=(Bi_groups[group],Bf_groups[group])
            
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2_3_phi_1_2.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")

    if (fitting and fit_phi):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)
        
    def process_fit(k):
        # print(foldername[k])
        nowf=datetime.now()
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
        # fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results_bcr_1_2_phi.mpa',skiprows=1)
        # diff_eff = diff_eff[diff_eff[:,0]<=0]
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
        def fit_func(x, bcr1, bcr2, mu1, sigma, tau, x00, zeta0, phi):
            x=diff_eff[:,0]+x00
            phi=phi*pi
            d=d0/np.cos((tilt[k]+zeta0)*rad)
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
            a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
            # plt.plot(a)
            # plt.savefig('a.eps', format='eps')
            th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
            S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
            eta=S.copy().real
            eta_aus=eta.copy()
            sum_diff = np.zeros(len(th))
            for l in range(len(wl)):
                lam=wl[l] #single wavelenght in micrometers
                b=2*pi/lam#beta value 
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
            eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
            x_int=np.arange(th[0],th[-1], 1e-6)
            for i in range(n_diff*2+1):
                f_int = interp1d(th,eta[i,:], kind="cubic")
                conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
                f_int = interp1d(x_int,conv, kind="cubic")
                eta_ang[i,:]=f_int(x*rad)
            aaa=eta_ang[n_diff-2:n_diff+3].ravel()
            return aaa
        P0= np.zeros(8) #fit_res[0]  # [*fit_res[0],0  # fit_res[0] # [*fit_res[0,:-1],0,0]  # fit_res[0] #  [8, 2,0, 2.01e-3, pi,0, 75, 1000, 0.0004] #    [5,0,2.6e-3] # 
        if (fitting):
            P0[0]=8
            P0[1]=1.
            P0[2]=3.5e-3
            P0[3]=0.0002
            P0[4]=5
            P0[5]=0
            P0[6]=0
            P0[7]=1
            Bi_groups=[[5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0],
                      [5, 0, 2.5e-3, 2e-5, 0.1, -0.0005/rad, -2, 0]]
            Bf_groups=[[10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2, 2],
                      [10, 5, 4e-3, 1.5e-3, 10, 0.0005/rad, 2, 2]]
            B=(Bi_groups[group],Bf_groups[group])
            for i in range(len(B[0])):
                if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                    P0[i]=(B[1][i]+B[0][i])/2
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            # fig=plt.figure()
            # ax1=fig.add_subplot()
            # ax1.errorbar(range(len(ff)), ff,yerr=fferr)
            # plt.show()
            xx=np.zeros(len(diff_eff[:,0])*5)
            xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
            try:
                for i in range(1):
                    p,cov=fit(fit_func,xx,ff, p0=P0,bounds=B)
                    P0=p
                    print(p)
            except RuntimeError:
                print("Error: fit not found")
            print(p)
            print(np.diag(cov)**0.5)
            now1f=datetime.now()
            print("fit time "+foldername[k]+"=",now1f-nowf)
            if (save_fit_res):
                with open(data_analysis+foldername[k]+'_fit_results_bcr_1_2_phi.mpa', 'w') as f:
                    np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")

    if (fitting and fit_phi):
        now=datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit,krange)#len(foldername)))
        now1=datetime.now()
        print("fit time=",now1-now)