#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:24:05 2022

@author: aaa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.stats import exponnorm
plt.rcParams.update(plt.rcParamsDefault)
pi=np.pi
rad=pi/180
fit_name="bcr_1_2_phi_no_zeta_0"
p_name=["$b_c \Delta\\rho_1$", "$\\theta_0$"]
p_units=[" $1/\mu m^2$", " nm", "  deg"]

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
    allwl_plots[g]+="WL distributions/"
    alldiff_eff_fit[g]+="Diff. effs + fits/"
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
tilt=np.array([0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52],dtype=float)
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

mu1 = 3.37493766e-03 
tau=4.15406881e+00
sigma = 1.93789263e-04

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
measur_groups=np.array([[0,2,3,4,5],[6,7,8,9,10,11,12],[1], range(13)])

for group in [3]: #0 for Juergen, 1 for Martin, 2 for Christian, 3 for all
    krange=np.array(measur_groups[group])
    krange=krange[np.argsort(tilt[krange])]
    def k_jz(theta, j, G,b):
        k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
        return k_jz
    def dq_j (theta, j, G,b):
        return b*np.cos(theta) - k_jz(theta, j, G, b)
    pendel_data=np.zeros((13,4))
    pendel_theta= np.zeros((13,2))
    pendel_fit=np.zeros((13,2))
    wlp=1e-2
    for k in krange:
        print(foldername[k])
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        diff_eff =  np.loadtxt(data_analysis+foldername[k]+"_diff_eff_new.mpa",skiprows=1)
        data_analysis1 = sorted_fold_path+foldername[2]+"/Data Analysis/"
        fit_res =  np.loadtxt(data_analysis1+foldername[2]+"_fit_results_"+fit_name+".mpa",skiprows=1)
        p=fit_res[0]
        diff_eff=diff_eff[diff_eff[:,0]<0]
        pendel_theta[k,0]=diff_eff[:,0][diff_eff[:,4]==np.amax(diff_eff[:,4])]
        pendel_theta[k,1]=diff_eff[:,0][diff_eff[:,2]==np.amax(diff_eff[:,2])]
        pendel_data[k,0]=diff_eff[:,4][diff_eff[:,4]==np.amax(diff_eff[:,4])]
        pendel_data[k,1]=diff_eff[:,5][diff_eff[:,4]==np.amax(diff_eff[:,4])]
        pendel_data[k,2]=diff_eff[:,2][diff_eff[:,2]==np.amax(diff_eff[:,2])]
        pendel_data[k,3]=diff_eff[:,3][diff_eff[:,2]==np.amax(diff_eff[:,2])]
        diff_eff_fit=np.transpose(diff_eff[:,2::2])
        diff_eff_err=diff_eff[:,3::2]
        thx=np.linspace(-0.5,0,len(diff_eff[:,0]))
        def plot_func(x, bcr1, bcr2, mu1, sigma, tau, x00,phi):
            x=thx+x00
            phi*=pi
            d=d0/np.cos((tilt[k])*rad)
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
                eta_ang[i,:]=f_int(x*rad)
            return eta_ang
        eta=plot_func(thx, *p)
        thx*=rad
        pendel_fit[k,0]=np.amax(eta[n_diff-1,:])
        pendel_fit[k,1]=np.amax(eta[n_diff-2,:])
    fig = plt.figure(figsize=(6,3))
    ax=fig.add_subplot(111)
    # print(tilt[krange])
    #pendel_theta*=rad
    # plt.plot(tilt[krange], pendel_theta[:,0])
    # plt.plot(tilt[krange], pendel_theta[:,1])
    ax.plot(tilt[krange], pendel_fit[krange,0], "-k", label= "Fit results order -1")
    ax.plot(tilt[krange], pendel_fit[krange,1],"-", color=(0.8,0,0), label= "Fit results order -2")
    ax.errorbar(tilt[krange], pendel_data[krange,0], yerr=pendel_data[krange,1], fmt="^k", label= "Data order -1")#color=(0.5,0.5,0.5),
    ax.errorbar(tilt[krange], pendel_data[krange,2], yerr=pendel_data[krange,3], fmt="^", color=(0.8,0,0), label="Data order -2")
    ax.set_xlabel("$\zeta$ (deg)")
    ax.set_ylabel("Max. diffraction efficiencies")
    ax.legend(loc=6)
    # plt.savefig("Pendell_"+fit_name+".pdf", format="pdf",bbox_inches="tight")
    
plt.show()

