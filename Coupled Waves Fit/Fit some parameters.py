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
tau0=0.001	#+/-	147.471394720765
mu0=2.5e-3#0.004632543663155012	#+/-	5.46776175965519e-05
sigma0=0.0005
M = mu0+tau0
sigma1=(sigma0**2+tau0**2)**0.5
##############################################################################

"""
Angular distribution: Gaussian
"""
div=0.00062
def ang_gauss(x):
    sig=div
    return 1/((2*pi)**0.5*sig)*np.exp(-x**2/(2*sig**2))
gx=np.arange(-3*div,3*div, 1e-6)
gauss_conv = ang_gauss(gx)/sum(ang_gauss(gx))

##############################################################################

n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves

LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM
bcr1=5.0 #scattering lenght x density
bcr2=0.
bcr3=0
n_0 =1.
phi=0
phi1=0
d0=78
krange=range(len(foldername))#[0,2,3,4,5] #[0]#np.arange(len(foldername))#

def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)
fitting=0
plotting=1
save_fit_res=0
wlpoints=50
wlp=5e-9
def process_fit(k):
    # print(foldername[k])
    nowf=datetime.now()
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff.mpa',skiprows=1)
    data_analysis1 = sorted_fold_path+foldername[1]+"/Data Analysis/"
    fit_res =  np.loadtxt(data_analysis1+foldername[1]+'_fit_results.mpa',skiprows=1)
    # diff_eff = diff_eff[diff_eff[:,0]<=0]
    diff_eff_aus=diff_eff.copy()
    for i in range(len(diff_eff[:,0])): 
        s=sum(diff_eff[i,2::2])
        diff_eff[i,2:]=diff_eff[i,2:]/s
    diff_eff_fit=diff_eff[:,2::2].copy()
    diff_eff_fit=np.transpose(diff_eff_fit)
    def fit_func(x, bcr1, bcr2, mu1,sigma, tau, x00,d):
        # tau=M-mu1
        # sigma=(sigma1**2-tau0**2)**0.5
        lambda_par=1/tau
        sigma1=(sigma**2+tau**2)**0.5
        x=diff_eff[:,0]+x00
        # d=d0/np.cos((tilt[k]+zeta0)*rad)
        d=d/np.cos((tilt[k])*rad)
        wl=np.linspace(mu1-2.5*sigma, mu1+1/lambda_par+3.5*sigma1, 10000)
        a = rho(wl,lambda_par, mu1, sigma)/sum(rho(wl,lambda_par, mu1, sigma))
        spl = UnivariateSpline(wl, a, k=4, s=0)
        I=spl.antiderivative()(wl)
        y=np.arange(I[I==np.amin(I)],I[I==np.amax(I)]+wlp,  wlp)
        xp=np.zeros(len(y))
        for i in range(len(y)):
            aus =abs(spl.antiderivative()(wl)-y[i])
            xp[i]=wl[aus==np.amin(aus)]
        wl=xp.copy()
        a=rho(xp,lambda_par, mu1, sigma)/sum(rho(xp,lambda_par, mu1, sigma))
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
        plt.plot(aaa)
        plt.savefig('ff.eps', format='eps')
        return aaa
    P0=  fit_res[0] # np.zeros(7) # fit_res[0] # [*fit_res[0,:-1],0,0]  # fit_res[0] #  [8, 2,0, 2.01e-3, pi,0, 75, 1000, 0.0004] #    [5,0,2.6e-3] # 
    # P0[0]=5
    # P0[1]=3
    # P0[2]=3e-3
    # P0[3]=0.00055
    # P0[4]=0.0005
    # P0[5]=0
    # P0[6]=78
    if (fitting):
        B=([4, 0, 2e-3, 1e-4, 0.001, -0.0005/rad, 70],[10, 2, 3.2e-3, 1e-3, 0.002, 0.0005/rad, 100])
        for i in range(len(B[0])):
            if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                P0[i]=(B[1][i]+B[0][i])/2
        ff=diff_eff_fit.ravel()
        fferr=(np.transpose(diff_eff[:,3::2])).ravel()
        fferr[fferr==0]=1e-8
        xx=np.zeros(len(diff_eff[:,0])*5)
        xx[0:len(diff_eff[:,0])]=diff_eff[:,0]
        #plt.plot(ff,"k")
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
            with open(data_analysis+foldername[k]+'_fit_results.mpa', 'w') as f:
                np.savetxt(f,(p,np.diag(cov)**0.5), header="bcr1 bcr2 mu phi thickness", fmt="%.6f")

if (fitting):
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    if __name__=="__main__":
        pool=Pool()
        pool.map(process_fit,krange)#len(foldername)))
    now1=datetime.now()
    print("fit time=",now1-now)
if (plotting):
    for k in krange:
        if (not fitting):
            now1=datetime.now()
        print(foldername[k])
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff.mpa',skiprows=1)
        fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results.mpa',skiprows=1)
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
        def plot_func(x, bcr1, bcr2, mu1, sigma, tau, x00,d):
            # tau=M-mu1
            # sigma=(sigma1**2-tau0**2)**0.5
            lambda_par=1/tau
            sigma1=(sigma**2+tau**2)**0.5
            x=diff_eff[:,0]+x00
            # d=d0/np.cos((tilt[k]+zeta0)*rad)
            d=d/np.cos((tilt[k])*rad)
            wl=np.linspace(mu1-2.5*sigma, mu1+1/lambda_par+3.5*sigma1, 10000)
            a = rho(wl,lambda_par, mu1, sigma)/sum(rho(wl,lambda_par, mu1, sigma))
            spl = UnivariateSpline(wl, a, k=4, s=0)
            I=spl.antiderivative()(wl)
            y=np.arange(I[I==np.amin(I)],I[I==np.amax(I)]+wlp,  wlp)
            xp=np.zeros(len(y))
            for i in range(len(y)):
                aus =abs(spl.antiderivative()(wl)-y[i])
                xp[i]=wl[aus==np.amin(aus)]
            wl=xp.copy()
            a=rho(xp,lambda_par, mu1, sigma)/sum(rho(xp,lambda_par, mu1, sigma))
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
            return eta_ang
        # print(wlpoints)
        thx=diff_eff[:,0]*rad
        eta=plot_func(diff_eff[:,0], *p)
        #print("here")
        # bbb=eta.ravel()
        # plt.plot(bbb)
        fig, ax = plt.subplots(3,figsize=(10,10))
        ax[0].set_title(foldername[k])
        ax[0].errorbar(diff_eff[:,0]*rad,diff_eff_fit[2,:], fmt="^k",  yerr=diff_eff[:,7], label="Data")
        ax[0].plot(thx,eta[n_diff,:],"--k", label="Fit")
        #ax[0].set_ylim([np.amin(diff_eff_fit[2,:])-0.4,np.amax(diff_eff_fit[2,:])])
        #ax[0].legend(loc=(5))
        for i in range(1,3):
            if i<3:
                #ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6-2*i],'o')
                ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6-2*i], fmt="^k", yerr=diff_eff[:,7-2*i], label="Data (-"+str(i)+")")
                #ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6+2*i],'o')
                ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6+2*i], fmt="v",  color = (0.8,0,0),  yerr=diff_eff[:,7+2*i],label="Data (+"+str(i)+")")
            ax[i].plot(thx,eta[n_diff-i,:],"--k", label="Fit (-"+str(i)+")")
            ax[i].plot(thx,eta[n_diff+i,:],"--",color = (0.8,0,0), label="Fit (+"+str(i)+")")   
            #ax[i].legend()
        # ax[n_diff+1].plot(th, sum_diff)
        # ax[n_diff+1].set_ylim([0.5,1.5])
        #   plt.errorbar(diff_eff[:,0],diff_eff[:,2*j+2],yerr=diff_eff[:,2*j+1],capsize=1)
        p_name=["$(b_c \\rho)_1$","$(b_c \\rho)_2$", "$\mu$", "$\sigma$","$\\tau$", "$x_0$","d"]
        p_units=[" $1/\mu m^2$"," $1/\mu m^2$"," nm", " nm", " nm", " deg", " $\mu m$"]
        text = "Fit results"
        for i in range(2,5):
            fit_res[0,i]*=1e3
            fit_res[1,i]*=1e3
        for i in range(len(p)):
            if not i%3:
                text+= "\n"
            else:
                text+= "\t"
            text+= p_name[i] + "=" + str("%.3f" % (fit_res[0,i],)) + "$\pm$" + str("%.3f" % (fit_res[1,i],)) + p_units[i]
        ax[-1].text(diff_eff[0,0]*rad,-np.amax(diff_eff_fit[0,:])*2/3, text,  bbox=dict(boxstyle="square", ec=(0, 0, 0), fc=(1,1,1)))
        #ax[1].text( diff_eff[0,0]*rad,np.amax(diff_eff_fit[3,:]), "p value="+str("%.3f" % (chi[1],)),  bbox=dict(boxstyle="square", ec=(0, 0, 0), fc=(1,1,1)))
        now2=datetime.now()
        print("plot time=",now2-now1)

duration = 0.2  # seconds
freq = 440  # Hz
for i in range (6):
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq+i%3*62))
    if i%3==2:
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
for i in range (2):
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq+2*62))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq+2*62+31))
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq+3*62+31))
    time.sleep(0.2)
  


"""
Merges fit results in a doc
"""
data_analysis = sorted_fold_path+foldername[0]+"/Data Analysis/"
fit_res =  np.loadtxt(data_analysis+foldername[0]+'_fit_results.mpa',skiprows=1)
tot_res = np.zeros((len(foldername), 8))
tot_cov=tot_res.copy()
for k in range(len(foldername)):
    #print(foldername[k])
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results.mpa',skiprows=1)
    tot_res[k,0]=tilt[k]
    tot_res[k,1:]=fit_res[0]
    tot_cov[k,0]=tilt[k]
    tot_cov[k,1:]=fit_res[1]
tot_res=tot_res[np.argsort(tot_res[:,0])]
tot_cov=tot_cov[np.argsort(tot_cov[:,0])]
print(tot_res)

with open(sorted_fold_path+'tot_fit_results.mpa', 'w') as f:
      np.savetxt(f,tot_res, header="tilt bcr1 bcr2 mu sigma tau x0 d", fmt="%.2f "+"%.6f "*len(fit_res[0,:]))
with open(sorted_fold_path+'tot_fit_covariances.mpa', 'w') as f:
      np.savetxt(f,tot_cov, header="tilt bcr1 bcr2 mu sigma tau x0 d", fmt="%.2f "+"%.6f "*len(fit_res[0,:]))

"""
Plot parameters evolution
"""
data_analysis = sorted_fold_path+foldername[2]+"/Data Analysis/"
fit_res =  np.loadtxt(sorted_fold_path+'tot_fit_results.mpa',skiprows=1)
fit_cov =  np.loadtxt(sorted_fold_path+'tot_fit_covariances.mpa',skiprows=1)
fig, ax = plt.subplots(len(fit_res[0,1:]),figsize=(10,10),sharex="col")
#plt.subplots_adjust(hspace=0.5)
plt.xticks(range(len(fit_res[:,0])),fit_res[:,0]) 

title=["bcr1","bcr2","mu", "sigma","tau", "x0","d"]
for i in range(len(fit_res[0,1:])):
    ax[i].set(ylabel=title[i])
    ax[i].errorbar(np.arange(len(foldername)),fit_res[:,i+1], yerr=fit_cov[:,i+1])
    ax[i].set_ylim([np.amin(fit_res[:,i+1])*(0.9),np.amax(fit_res[:,i+1])*(1.1)])
   
"""
"""
for k in krange:
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    fit_res =  np.loadtxt(data_analysis+foldername[k]+'_fit_results.mpa',skiprows=1)
    mu=fit_res[0,2]
    # tau=M-mu
    # sigma=(sigma1**2-tau0**2)**0.5
    # lambda_par=1/tau
    sigma=fit_res[0,3]
    tau=fit_res[0,4]
    lambda_par=1/tau
    sigma1=(sigma**2+tau**2)**0.5
    wl=np.linspace(mu-2.5*sigma,mu+1/lambda_par+3.5*sigma1, 10000)
    a = rho(wl,lambda_par, mu, sigma)/sum(rho(wl,lambda_par, mu, sigma))
    spl = UnivariateSpline(wl, a, k=3, s=0)
    d=spl.antiderivative()(wl)
    s=wlpoints
    y=np.arange(d[d==np.amin(d)],d[d==np.amax(d)]+wlp,  wlp)
    # print("points=",len(y))
    x=np.zeros(len(y))
    for i in range(len(y)):
        aus =abs(spl.antiderivative()(wl)-y[i])
        x[i]=wl[aus==np.amin(aus)]
    fig = plt.figure(figsize=(10,10))
    ax= fig.add_subplot()
    ax.set_title(foldername[k])
    #ax.plot(wl,d/np.amax(d))
    ax.plot(wl,a/np.amax(a), label= "WL distribution")
    #ax.plot(x,x*0,"k.")
    #ax.set_xlim([0,0.011])
    a=rho(x,lambda_par, mu, sigma)/sum(rho(x,lambda_par, mu, sigma))
    #ax.plot(x,a/np.amax(a),"g.")
    ax.vlines(x[a==np.amax(a)], 0,1, ls="dashed", label="$\lambda_{max}=$"+str(x[a==np.amax(a)]*1e3)+" nm")
    ax.legend()
    

# def format_axes(fig):
#     for i, ax in enumerate(fig.axes):
#         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)
#         ax.plot(range(10),range(10))
# fig = plt.figure(constrained_layout=True)

# gs = GridSpec(3, 3, figure=fig)
# ax1 = fig.add_subplot(gs[0, :])
# # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
# ax2 = fig.add_subplot(gs[1, :-1])
# ax3 = fig.add_subplot(gs[1:, -1])
# ax4 = fig.add_subplot(gs[-1, 0])
# ax5 = fig.add_subplot(gs[-1, -2])
# ax1.plot(range(10),range(10))
# fig.suptitle("GridSpec")
# format_axes(fig)

# plt.show()