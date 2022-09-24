#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:13:16 2022

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
fit_name="bcr_1_2_phi_no_zeta0"
n_shared=3
p_name=["$(b_c \\rho)_1$","$(b_c \\rho)_2$", "$\phi$", "$\mu$", "$\sigma$","$\\tau$", "$x_0$"]
p_units=[" $1/\mu m^2$"," $1/\mu m^2$", "$ \pi$"," nm", " nm", "", " deg"]

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
allfits_plots= [alldata_analysis + "All fits plots/Juergen/"+fit_name+"/",
                alldata_analysis + "All fits plots/Martin/"+fit_name+"/",
                alldata_analysis + "All fits plots/Christian/"+fit_name+"/",
                alldata_analysis + "All fits plots/All groups/"+fit_name+"/"]
alldiff_eff_fit=allfits_plots.copy()
allwl_plots=allfits_plots.copy()

for g in range(4):
    alldiff_eff_fit[g]+="Diff. effs + fits/"
    allwl_plots[g]+="WL distributions/"
    # if os.path.exists(allfits_plots[g]):
    #         shutil.rmtree(allfits_plots[g])
    # os.makedirs(allfits_plots[g])

    # if os.path.exists(alldiff_eff_fit[g]):
    #         shutil.rmtree(alldiff_eff_fit[g])
    # os.makedirs(alldiff_eff_fit[g])

    # if os.path.exists(allwl_plots[g]):
    #         shutil.rmtree(allwl_plots[g])
    # os.makedirs(allwl_plots[g])

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
This block caclulates the diffraction efficiencies n(x)= n_0 + n_1 cos(Gx)
"""
def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)

def diff_eff_calc(k,bcr1, bcr2, phi, mu1, sigma, tau, x00):
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+"_diff_eff_new.mpa",skiprows=1)
    x=diff_eff[:,0]+x00
    phi=phi*pi
    d=d0/np.cos((tilt[k]+zeta0)*rad)
    wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
    a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
    # plt.plot(a)
    # plt.savefig("a.eps", format="eps")
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
    with open(data_analysis+foldername[k]+"_multi_fit_func_aus_"+fit_name+".mpa", "w") as f:
        np.savetxt(f,aaa, header="Raveled func")
    with open(sorted_fold_path+"test"+fit_name+".mpa", "w") as f:
        np.savetxt(f,aaa, header="Raveled func")

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
zeta0=0
B0i=[5, 0, 0, 1.4e-3, 2e-5, 0.5, -0.0005/rad, -2]
B0f=[13, 7,2 , 4e-3, 0.65e-3, 10, 0.0005/rad, 2]
Bi_groups=[B0i, B0i, B0i, B0i]
Bf_groups=[B0f, B0f, B0f, B0f]
measur_groups=[[0,2,3,4,5],[6,7,8,9,10,11,12],[1], range(13)]

for group in [2]: #0 for Juergen, 1 for Martin, 2 for Christian, 3 for all
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time group "+str(group)+" = ", current_time)
    tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
    plt.rcParams["font.size"] = 18
    plt.rcParams["lines.markersize"] = 12
    plt.rcParams["lines.linewidth"] = 4
    # plt.rcParams.update(plt.rcParamsDefault)
    fig_size=[20,18]
    krange=measur_groups[group]

    fitting=1
    plotting=1
    extended_plot=1
    save_fit_res=1
    wl_plot=1
    param_ev_plot=1
    close_fig=0
    wlp=1e-2
    def process_fit_aus(k):
        # print(foldername[k])
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        PAR=np.loadtxt(data_analysis+foldername[k]+"_multi_fit_PAR_"+fit_name+".mpa",skiprows=1)
        # print(PAR)
        diff_eff_calc(k,*PAR)
    
    def fit_func(x, FP):
        L=(len(FP)-n_shared)//len(krange)
        # print(L)
        kaus=0
        for kf in krange:
            data_analysis = sorted_fold_path+foldername[kf]+"/Data Analysis/"
            P=[*FP[0:n_shared],*FP[n_shared+kaus*L:n_shared+L*(1+kaus)]]
            # print(P)
            with open(data_analysis+foldername[kf]+"_multi_fit_PAR_"+fit_name+".mpa", "w") as f:
                np.savetxt(f,P, header="PAR")
            kaus+=1
        if __name__=="__main__":
            pool=Pool()
            pool.map(process_fit_aus,krange)
        AAA=np.array([])
        for kf in krange:
            data_analysis = sorted_fold_path+foldername[kf]+"/Data Analysis/"
            aaa=np.loadtxt(data_analysis+foldername[kf]+"_multi_fit_func_aus_"+fit_name+".mpa", skiprows=1)
            AAA=np.append(AAA, aaa)
        # plt.plot(AAA)
        return AAA
        
    if fitting:
        FF=np.array([])
        FFERR=np.array([])
        P0=np.zeros(7) 
        P0[0]=8
        P0[1]=1.
        P0[2]=1
        P0[3]=3.5e-3
        P0[4]=0.0002
        P0[5]=5
        P0[6]=0

        Paus=P0.copy()
        BI=Bi_groups[group]
        BF=Bf_groups[group]
        BIaus=BI.copy()
        BFaus=BF.copy()
        kaus=0
        for kf in krange:
            data_analysis = sorted_fold_path+foldername[kf]+"/Data Analysis/"
            diff_eff =  np.loadtxt(data_analysis+foldername[kf]+"_diff_eff_new.mpa",skiprows=1)
            diff_eff_fit=diff_eff[:,2::2].copy()
            diff_eff_err=diff_eff[:,3::2].copy()
            ff=np.transpose(diff_eff_fit).ravel()
            fferr=(np.transpose(diff_eff_err)).ravel()
            FF=np.append(FF,ff)
            FFERR=np.append(FFERR,fferr)
            if kaus>0:
                P0=np.append(P0, Paus[n_shared:])
                BI=np.append(BI,BIaus[n_shared:])
                BF=np.append(BF,BFaus[n_shared:])
            kaus+=1
        # print(len(P0))
        # plt.plot(FF)
        # print(len(P0))
        B=(BI,BF)
        for i in range(len(B[0])):
            if (P0[i]<B[0][i] or P0[i]>B[1][i]):
                P0[i]=(B[1][i]+B[0][i])/2
        def wrapper_fit_func(x, *args):
            return fit_func(x, args)
        nowf=datetime.now()
        p,cov=fit(wrapper_fit_func,range(len(FF)),FF, p0=P0,bounds=B)
        nowf1=datetime.now()
        print(p)
        print("fit time=",nowf1-nowf)
        if (save_fit_res):
            L=(len(P0)-n_shared)//len(krange)
            # print(L)
            kaus=0
            for kf in krange:
                data_analysis = sorted_fold_path+foldername[kf]+"/Data Analysis/"
                P=[*p[0:n_shared],*p[n_shared+kaus*L:n_shared+L*(1+kaus)]]
                c=np.diag(cov)**0.5
                COV=[*c[0:n_shared],*c[n_shared+kaus*L:n_shared+L*(1+kaus)]]
                # print(P)
                with open(data_analysis+foldername[kf]+"_group_"+str(group)+"_multi_fit_PAR_"+fit_name+".mpa", "w") as f:
                    np.savetxt(f,(P,COV), header="PAR")
                kaus+=1
        
    if (plotting):
        for k in krange:
            now1=datetime.now()
            print(foldername[k])
            data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
            diff_eff =  np.loadtxt(data_analysis+foldername[k]+"_diff_eff_new.mpa",skiprows=1)
            fit_res =  np.loadtxt(data_analysis+foldername[k]+"_group_"+str(group)+"_multi_fit_PAR_"+fit_name+".mpa", skiprows=1)#np.loadtxt(data_analysis+foldername[k]+"_fit_results_"+fit_name+".mpa",skiprows=1)
            p=fit_res[0]
            print(p)
            def plot_func(x, bcr1, bcr2, phi, mu1, sigma, tau, x00):
                x=diff_eff[:,0]+x00
                phi*pi
                d=d0/np.cos((tilt[k]+zeta0)*rad)
                wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
                a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
                th=[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,2*len(x))#np.linspace(x[0]*rad-3*div,x[-1]*rad+3*div,3*len(x))#
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
            thx=diff_eff[:,0]*rad
            eta=plot_func(diff_eff[:,0], *p)
            print(eta[n_diff,:])
            text = "Fit results"
            fig = plt.figure(figsize=(fig_size[0],fig_size[1]))#constrained_layout=True
            gs_t = GridSpec(5, 2, figure=fig,hspace=0, top=0.95)
            gs_b =GridSpec(5, 2, figure=fig, wspace=0)
            ax = [fig.add_subplot(gs_t[0,:]), 
                  fig.add_subplot(gs_t[1,:]),
                  fig.add_subplot(gs_t[2,:]),
                  fig.add_subplot(gs_b[3, 0]),
                  fig.add_subplot(gs_b[3, 1])]
            for i in range(len(ax)):
                if i!=2 and i!=4:
                    ax[i].tick_params(axis="x", labelbottom=False, bottom = False)
                if i>2:
                    ax[i].tick_params(axis="y", labelleft=False, left = False)
            ax[0].set_title(foldername[k])
            ax[0].errorbar(diff_eff[:,0]*rad,diff_eff[:,6], fmt="^k",  yerr=diff_eff[:,7], label="Data")
            ax[0].plot(thx,eta[n_diff,:],"--k", label="Fit")
            for i in range(1,3):
                if i<3:
                    ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6-2*i], fmt="^k", yerr=diff_eff[:,7-2*i], label="Data (-"+str(i)+")")
                    ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6+2*i], fmt="v",  color = (0.8,0,0),  yerr=diff_eff[:,7+2*i],label="Data (+"+str(i)+")")
                ax[i].plot(thx,eta[n_diff-i,:],"--k", label="Fit (-"+str(i)+")")
                ax[i].plot(thx,eta[n_diff+i,:],"--",color = (0.8,0,0), label="Fit (+"+str(i)+")")   
                #ax[i].legend()
            mu=p[3]
            sigma=p[4]
            tau=p[5]
            wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu, scale=sigma)
            a = rho(wl,tau, mu, sigma)/sum(rho(wl,tau, mu, sigma))
            ax[-1].plot(wl,a/np.amax(a), label= "WL distribution")
            ax[-1].vlines(wl[a==np.amax(a)], 0,1, ls="dashed", label="$\lambda_{max}=$"+str("%.3f" % (wl[a==np.amax(a)]*1e3),)+" nm")
            mean=exponnorm.ppf(0.5,K=tau, loc=mu, scale=sigma)
            ax[-1].vlines(mean, 0,a[abs(wl-mean)==np.amin(abs(wl-mean))]/np.amax(a), ls="dashdot", label="$\lambda_{mean}=$"+str("%.3f" % (mean*1e3),)+" nm")
            ax[-1].legend(loc=1)
            for i in range(3,5):
                fit_res[0,i]*=1e3
                fit_res[1,i]*=1e3
            for i in range(len(p)):
                if not i%2:
                    text+= "\n"
                else:
                    text+= "\t"
                text+= p_name[i] + "=" + str("%.3f" % (fit_res[0,i],)) + "$\pm$" + str("%.3f" % (fit_res[1,i],)) + p_units[i]
            ax[-2].text(0.5,0.5,text,va="center", ha="center")
            if close_fig:
                plt.close(fig)
            now2=datetime.now()
            print("plot time=",now2-now1)
    
    duration = 0.2  # seconds
    freq = 440  # Hz
    for i in range (6):
        os.system("play -nq -t alsa synth {} sine {}".format(duration, freq+i%3*62))
        if i%3==2:
            os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))
    for i in range (2):
        os.system("play -nq -t alsa synth {} sine {}".format(duration, freq+2*62))
        os.system("play -nq -t alsa synth {} sine {}".format(duration, freq+2*62+31))
        os.system("play -nq -t alsa synth {} sine {}".format(duration, freq+3*62+31))
        time.sleep(0.2)
      
    # """
    # Merges fit results in a doc
    # """
    # nmeas_groups=[5,7,1,13]
    # tot_res = np.zeros((2*nmeas_groups[group], 9))
    # kaus=-1
    # for k in krange:
    #     kaus+=1
    #     #print(foldername[k])
    #     data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    #     fit_res =  np.loadtxt(data_analysis+foldername[k]+"_fit_results_"+fit_name+".mpa",skiprows=1)
    #     tot_res[kaus,0]=tilt[k]
    #     tot_res[kaus,1:]=fit_res[0]
    #     tot_res[kaus+1,1:]=fit_res[1]    
    # with open(allfits_plots[group]+"tot_fit_results_"+fit_name+".mpa", "w") as f:
    #       np.savetxt(f,tot_res, header="tilt bcr1 bcr2 mu sigma tau x0 d", fmt="%.2f "+"%.6f "*len(fit_res[0,:]))
    # if group==3:
    #     with open(sorted_fold_path+"Total results/tot_fit_results_"+fit_name+".mpa", "w") as f:
    #           np.savetxt(f,tot_res, header="tilt bcr1 bcr2 mu sigma tau x0 d", fmt="%.2f "+"%.6f "*len(fit_res[0,:]))

    
    # """
    # Plot parameters evolution
    # """
    # B=[Bi_groups[group],Bf_groups[group]]
    # if param_ev_plot:
    #     if group!=2:
    #         fit_res =  np.loadtxt(allfits_plots[group]+"tot_fit_results_"+fit_name+".mpa",skiprows=1)
    #         fig, ax = plt.subplots(len(fit_res[0,1:]),figsize=(fig_size[0],fig_size[1]),sharex=True)
    #         #plt.subplots_adjust(hspace=0.5)
    #         plt.xticks(range(len(fit_res[:,0])),fit_res[:,0]) 
    #         for i in range(len(fit_res[0,1:])):
    #             ax[i].set_ylabel(p_name[i])
    #             ax[i].errorbar(np.arange(len(fit_res[::2,i+1])),fit_res[::2,i+1], yerr=fit_res[1::2,i+1])
    #             ax[i].set_ylim(B[0][i],B[1][i])
    #         plt.savefig(allfits_plots[group]+"Param_evolution_"+fit_name+".png", format="png",bbox_inches="tight")
    #         if close_fig:
    #             plt.close(fig)

    # """
    # This block plots the wl distributions
    # """
    # plt.rcParams["font.size"] = 11
    # plt.rcParams["lines.markersize"] = 6
    # plt.rcParams["lines.linewidth"] = 1
    # if wl_plot:
    #     tilt=np.array(tilt)
    #     krange=np.array(krange)
    #     krange=krange[np.argsort(tilt[krange])]
    #     # print(krange)
    #     group_names=["Jürgen's","Martin's","Christian's", "All"]
    #     fit_res_meas =  np.loadtxt(allfits_plots[group]+"tot_fit_results_"+fit_name+".mpa",skiprows=1)[::2,:]
    #     n_plots=4
    #     if group==2:
    #         n_plots=2
    #     kaus=-1
    #     tilt0=str(tilt[krange[0]])
    #     for k in krange:
    #         kaus+=1
    #         if kaus%n_plots==0:
    #             tilt1=str(tilt[krange[kaus-1]])
    #             if kaus>0:
    #                 plt.savefig(allwl_plots[group]+tilt0+"deg-"+tilt1+"deg_wl.png", format="png",bbox_inches="tight")
    #                 if close_fig:
    #                     plt.close(fig)
    #                 tilt0=str(tilt[k])
    #             fig, ax = plt.subplots(n_plots,figsize=(8.27,11.69), dpi=100, sharex=False)
    #             ax[0].set_title(group_names[group]+" measurements")
    #         data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    #         fit_res =  np.loadtxt(data_analysis+foldername[k]+"_fit_results_"+fit_name+".mpa",skiprows=1)
    #         mu=fit_res[0,2]
    #         sigma=fit_res[0,3]
    #         tau=fit_res[0,4]
    #         wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu, scale=sigma)
    #         x=np.linspace(wl[0],wl[-1],10000)
    #         a=rho(wl,tau, mu, sigma)/sum(rho(wl,tau, mu, sigma))
    #         # ax[kaus%n_plots].plot(wl,a/np.amax(a),"k.", label= "WL distribution")
    #         mean=exponnorm.ppf(0.5,K=tau, loc=mu, scale=sigma)
    #         ax[kaus%n_plots].vlines(mean, 0,a[abs(wl-mean)==np.amin(abs(wl-mean))]/np.amax(a), ls="dashdot", label="$\lambda_{mean}=$"+str("%.3f" % (mean*1e3),)+" nm")
    #         a=rho(x,tau, mu, sigma)/sum(rho(x,tau, mu, sigma))
    #         ax[kaus%n_plots].plot(x,a/np.amax(a),"k-", label= "WL distribution\n$\zeta = $"+str(tilt[k])+" deg")
    #         ax[kaus%n_plots].vlines(x[a==np.amax(a)][0], 0,1, ls="dashed", label="$\lambda_{max}=$"+str("%.3f" % (x[a==np.amax(a)]*1e3),)+" nm")
    #         ax[kaus%n_plots].legend(loc=1, fontsize=13)
    #         ax[kaus%n_plots].set_xlim([0,12e-3])
    #         if k==krange[-1]:
    #             plt.savefig(allwl_plots[group]+tilt0+"deg-"+str(tilt[k])+"deg_wl.png", format="png",bbox_inches="tight")
    #             if close_fig:
    #                 plt.close(fig)
        
