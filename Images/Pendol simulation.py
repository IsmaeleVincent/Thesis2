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

n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves
lam=3e-3
LAM= 0.5 #grating constant in micrometers
G=2*pi/LAM
bcr1=8.0 #scattering lenght x density
bcr2=0
bcr3=0
n_0 =1.
phi=0.
phi1=0.
d0=0
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
tilt=np.linspace(0,500,1000)#np.sort(tilt)
pendol = np.zeros((len(tilt),5))
pendol[:,0] = tilt
k=-1
for zeta in tilt:
    k+=1
    d=zeta#d0/np.cos((zeta*rad))
    # pendol[k,0] = d
    th=[-0.00459184]#np.linspace(-0.015,0,50)#[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]
    S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
    eta=S.copy().real
    eta_aus=eta.copy()
    sum_diff = np.zeros(len(th))
    b=2*pi/lam #beta value 
    n_1 = bcr1*2*pi/b**2
    n_2 = bcr2*2*pi/b**2
    n_3 = bcr3*2*pi/b**2
    for t in [0]:#range(len(th2)):
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
            eta[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
        sum_diff[t] = sum(eta[:,t])
    pendol[k,1]=eta[n_diff,:]
    pendol[k,2]=eta[n_diff-1,:]
    pendol[k,3]=eta[n_diff-2,:]
    pendol[k,4]=eta[n_diff-3,:]
plt.plot(pendol[:,0],pendol[:,1],"-k", label="Order 0")   
plt.plot(pendol[:,0],pendol[:,2],"--k", label="Order 1")
plt.plot(pendol[:,0],pendol[:,3],"--", color=(0.8,0,0),label="Order 2")
plt.plot(pendol[:,0],pendol[:,4],"--", color=(0,0,0.5), label="Order 3")
plt.legend()
plt.xlabel("Thickness (arb)")
plt.ylabel("Diff. efficiency")