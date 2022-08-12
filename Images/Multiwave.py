 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:09:28 2022

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
import math
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
tilt=[0,40,48,61,69,71,79,80,81,79,79,79,79]
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measurement
"""
This block fits the diffraction efficiencies n(x)= n_0 + n_1 cos(Gx)
"""

n_diff= 2 #number of peaks for each side, for example: n=2 for 5 diffracted waves
lam= 4e-3 #incoming wavelenght in micrometers
LAM= 0.5 #grating constant in micrometers
b=2*pi/lam #beta value 
G=2*pi/LAM
bcr1=7.0#scattering lenght x density
bcr2=3
n_0 =1.00
n_1 = bcr1*2*pi/b**2 
#print(n_1)
phi =0
def k_jz(theta, j, G):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G):
    return b*np.cos(theta) - k_jz(theta, j, G)
for k in range(1,2):#len(foldername)):
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff.mpa',skiprows=1)    
    fig, ax = plt.subplots(n_diff+1,figsize=(9,5),sharex=True)
    fig.text(0, 0.5, 'Counts', va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0.1)
    ax[0].tick_params('x', bottom=False)
    ax[1].tick_params('x', bottom=False)
    ax[0].set_title("Example of diffracted waves intensities")
    ax[0].plot(diff_eff[:,0]*rad,diff_eff[:,6], "^-k", label="Diff. order 0")
    ax[0].legend(loc=8)
    for i in range(1,3):
        ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6-2*i], "^-k", label="Diff. order -"+str(i))
        ax[i].plot(diff_eff[:,0]*rad,diff_eff[:,6+2*i], "^-",  color = (0.8,0,0),  label="Diff. order +"+str(i))
        ax[i].legend(loc=9)
    ax[-1].set_xlabel("Incidence angle (rad)")
plt.tight_layout()
plt.savefig("Multiwave.eps", format='pdf')