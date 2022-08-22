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


##############################################################################
krange=range(len(foldername))#[0,2,3,4,5]#range(7,len(foldername))#[0,2,3,4,5] #np.arange(len(foldername))#

pendol = np.zeros((len(foldername),3))
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

print(pendol)
pendol[:,0]=pendol[np.argsort(pendol[:,0],axis=0),0]
pendol[:,1]=pendol[np.argsort(pendol[:,0],axis=0),1]
print(pendol)
plt.errorbar(pendol[:,0],pendol[:,1],pendol[:,2], fmt="^--k")

