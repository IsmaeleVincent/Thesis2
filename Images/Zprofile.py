#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:45:54 2022

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
import socket
import shutil
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare as cs
import math
pi=np.pi
rad=pi/180
plt.rcParams["font.size"] = 15
sorted_fold_path="/home/aaa/Desktop/Thesis/Script/Trial/Sorted data/" #insert folder of sorted meausements files
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
tilt=[0,40,48,61,69,71,77.88,76.76,75.64,74.52,79,80,81]
n_theta=[26,46,28,17,16,20,48,43,59,24,21,20,19]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measureme0nt
foldername=np.array(foldername)
foldername=foldername[np.argsort(foldername)]
mins=[117,204,307,335,339]
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
f="Zprofile.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
print(data[:,0][data[:,1]==np.amin(data[300:310,1])])
ax.set_xlabel("Meauserement number")
ax.set_ylabel("Avg counts")
ax.set_xlim([0,400])
# ax.plot(xplt-xmax, gauss(xplt,*p[0:3]), "r-", label="0th order")
# ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "g-", label="1st order")
# ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "b-", label="2nd order")
ax.plot(data[:sum(n_theta),0],data[:sum(n_theta),1], "k--", label="Measurement")
# plt.legend(loc=0)
# ax.set_xlim([0,sum(n_theta)])
ax.set_ylim([0,2.5])
for i in mins:
    ax.arrow(data[i,0], 0.1, 0, data[i,1]-0.35, head_width=5, head_length=0.1, color="k")
a=-1
for i in range(len(n_theta)):
   s= sum(n_theta[0:i]) +2 #n_theta[i]//4
   if i==10:
    ax.text(data[s,0], data[s+3,1]-0.05-2*a*0.15, "$\zeta$="+str(tilt[i])+"$^o$", color="k", fontsize=12) 
   else:
       if (i>6):
           s+=2
       ax.text(data[s,0], data[s+6,1]-0.05+a*0.15, "$\zeta$="+str(tilt[i])+"$^o$", color="k", fontsize=12) 
       a=-a

plt.savefig('Dips.eps', format='eps',bbox_inches='tight')