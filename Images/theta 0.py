#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:07:46 2022

@author: aaa
"""
import os
import shutil
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare as cs
import math
from scipy.interpolate import interp1d
plt.rcParams.update(plt.rcParamsDefault)
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
allcorrectedfits = alldata_analysis + "All Corrected Fits/"
allcontroldiff = alldata_analysis + "All Control new diff/"
tiltangles=[0,40,48,61,69,71,79,80,81]
foldername=[]
for i in range(len(tiltangles)):
    foldername.append(str(tiltangles[i])+"deg")
foldername.append("79-4U_77c88deg")
foldername.append("79-8U_76c76deg")
foldername.append("79-12U_75c64deg")
foldername.append("79-16U_74c52deg")
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
step_theta=[0.03,0.03,0.05,0.05,0.05,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
n_pixel = 16384 #number of pixels in one measurement

krange=[1]

"""
This block calculates the diffraction intensities, first part estimates theta=0
and second part the diff eff for each theta
"""
plot=1
def gauss(x, A, x0,sx):
      return A/sx*np.exp(-(x-x0)**2/(2*(sx)**2))
for k in krange:#range(11,len(foldername)):#range(8,10):#
    print(k)
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    controldiff = data_analysis + "Control Diff/" 
    if os.path.exists(controldiff):
        shutil.rmtree(controldiff)
    os.makedirs(controldiff)
    matrixes = [np.loadtxt(sorted_fold_path+foldername[k]+"/Matrixes/"+foldername[k]+"_"+str("%03d" % (j,))+".mpa") for j in range (1,n_theta[k]+1)]
    stack = np.stack(matrixes,axis=2)
    xyzabsmax = np.where(stack[:,:,:]==np.amax(stack[:,:,:]))
    yabsmax = xyzabsmax[0][0]
    xabsmax = xyzabsmax[1][0]
    zabsmax = xyzabsmax[2][0]
    roi =  np.loadtxt(data_analysis+foldername[k]+'_ROI+Peaks.mpa',skiprows=1).astype(int)
    npeaks=roi[:,3:5]
    xpeaks=roi[:,5:7]
    data_and_fit  =  np.loadtxt(data_analysis+foldername[k]+'_fit+data.mpa',skiprows=1)
    diff_eff = np.zeros((len(stack[0,0,:]),12))
    #print(foldername[k])
    zprofile0 = np.zeros(len(stack[0,0,:]))
    for z in range(len(stack[0,0,:])):
        zprofile0[z] = np.sum(stack[roi[0,0]:roi[-1,0], xabsmax-2:xabsmax+2,z])
    zmin1=roi[:,7][roi[:,0]==yabsmax]
    zmin2=roi[:,8][roi[:,0]==yabsmax]
    f2 = interp1d(np.where(zprofile0)[0], zprofile0, kind='cubic')
    zplt=np.linspace(0,len(zprofile0)-1, 10000)
    # if (k==6):
    #     zplt=np.linspace(4,len(zprofile0)-1, 10000)
    zplt1=np.linspace(zmin1,zmin2, 1000)
    zmax= zplt1[f2(zplt1)==np.amax(f2(zplt1))]
    print(zmax)
    theta=np.zeros(len(stack[0,0,:]))
    if(zmin1>0 and zmin2<len(stack[0,0,:])-1):
        zplt1=np.linspace(0,zmax, 1000)
        z1=zplt1[f2(zplt1)==np.amin(f2(zplt1))]
        zplt1=np.linspace(zmax,len(stack[0,0,:])-1, 1000)
        z2=zplt1[f2(zplt1)==np.amin(f2(zplt1))]
        c=(z1+z2)*0.5
    else:
        c= zplt1[f2(zplt1)==np.amax(f2(zplt1))]
        z1=zmin1
        z2=zmin2
    if (k==0):
        c=19 #for the ones in which there are not enough measurements I inserted the value manually
    if (k==4):
        c=12
    if (k==6):
        c=19.1
    if (k==7):
        c=16
    if (k==8):
        c=18
    if (k==12):
        c=23
    if(plot):    
        for z in range(len(stack[0,0,:])):
            theta[z]=(z-c)*step_theta[k]
        tplt=zplt#np.linspace(theta[0],theta[-1], 10000)*rad
        fig = plt.figure(figsize=(7,2), dpi=200)
        ax = fig.add_subplot(111)
        # ax.set_title(foldername[k])
        ax.set_ylabel("Counts (0th order)")
        ax.set_xlabel("Measurement number")
        # ax.axvline(zmax, color="b")
        ax.plot(tplt,f2(zplt), "k-", label="Splines\ninterpolation")
        zmax=np.arange(10000)[tplt==np.amin(abs(tplt))][0]
        print(zmax)
        # z1=tplt[f2(zplt)==np.amin(f2(zplt[:zmax]))]
        ax.axvline(z1, color=(0.8,0,0))
        # z2=tplt[f2(zplt)==np.amin(f2(zplt[zmax:]))]
        ax.axvline(z2, color=(0,0.5,0))
        # ax.text(z2, np.amax(zprofile0)," $z_+$", va="top",  color=(0,0.5,0))
        # ax.text(z1, np.amax(zprofile0)," $z_-$",  va="top", color=(0.8,0,0))
        ax.text(c, np.amax(zprofile0)," $\\theta=0$",  va="top", color=(0,0,0))
        # ax.text(z2, np.amax(zprofile0)," Second\n min. ", va="top",  color=(0,0.5,0))
        # ax.text(z1, np.amax(zprofile0)," First\n min.",  va="top", color=(0.8,0,0))
        ax.axvline(c, color="k", ls="dashed")
        ax.plot(zprofile0, "k^", label="Data")
        ax.legend(loc=4)
plt.show()
# plt.savefig("Theta0.pdf", format="pdf",bbox_inches="tight")