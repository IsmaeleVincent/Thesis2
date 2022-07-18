#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:37:48 2022

@author: aaa
"""

import os
import shutil
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare as cs
from scipy import stats
import math
import pandas as pd
from scipy.stats import poisson as ps
from scipy.stats import norm
from scipy.stats import chisquare
import matplotlib.ticker as ticker
from PIL import Image as im

sorted_fold_path="/home/aaa/Desktop/thesis_L1/Sorted data/" #insert folder of sorted meausements files
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
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
step_theta=[0.03,0.03,0.05,0.05,0.05,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
n_pixel = 16384 #number of pixels in one measurement


"""
This block calculates the diffraction efficiencies, first part estimates theta=0
and second part the diff eff for each theta
"""
def gauss(x, A, x0,sx):
    return A/sx*np.exp(-(x-x0)**2/(2*(sx)**2))
def distrm3(x,A0,A1,A2,x0,x1,x2,s0,s1,s2):
    return gauss(x,A0,-x0,s0)+gauss(x, A1,-x1,s1)+gauss(x, A2, -x2,s2)
def distrp3(x,A0,A1,A2,x0,x1,x2,s0,s1,s2):
    return gauss(x,A0,x0,s0)+gauss(x, A1, x1,s1)+gauss(x, A2, x2,s2)
def distr1(x, A, x0,sx):
    return A/sx*np.exp(-(x-x0)**2/(2*(sx)**2))

for k in range(0,1):#len(foldername)):
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    controlfits = data_analysis + "Control Fits/" 
    if os.path.exists(controlfits):
        shutil.rmtree(controlfits)
    os.makedirs(controlfits)
    matrixes = [np.loadtxt(sorted_fold_path+foldername[k]+"/Matrixes/"+foldername[k]+"_"+str("%03d" % (j,))+".mpa") for j in range (1,n_theta[k]+1)]
    stack = np.stack(matrixes,axis=2)
    xyzabsmax = np.where(stack[:,:,:]==np.amax(stack[:,:,:]))
    yabsmax = xyzabsmax[0][0]
    xabsmax = xyzabsmax[1][0]
    zabsmax = xyzabsmax[2][0]
    roi =  np.loadtxt(data_analysis+foldername[k]+'_ROI+Peaks.mpa',skiprows=1).astype(int)
    data_and_fit  =  np.loadtxt(data_analysis+foldername[k]+'_fit+data.mpa',skiprows=1)
    P0m = np.zeros(9)
    P0p = np.zeros(9)
    print(foldername[k])
    a=0
    for y in range(len(roi[:,0])):
        for z in range(len(stack[0,0,:])):
            if data_and_fit[z*len(roi[:,0])+y][1]>0:
                bckg = data_and_fit[z*len(roi[:,0])+y][2]
                for j in range(len(P0m)):
                    P0m[j]=data_and_fit[z*len(roi[:,0])+y][j+3] 
                    P0p[j]=data_and_fit[z*len(roi[:,0])+y][j+3+len(P0m)]
                data=np.zeros((roi[y][2]-roi[y][1]+1,2))
                data[:,0] =  np.arange(roi[y][1],roi[y][2]+1)
                data[:,1] =  stack[roi[y][0],roi[y][1]:(roi[y][2]+1),z]
                data[:,0] = (data[:,0]-xabsmax)
                # fig, ax = plt.subplots(2,figsize=(10,10))
                # ax[0].set_title(foldername[k] +'-Line ' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z)))
                if(P0m[2]>0 or P0m[2]>0):
                    # ax[0].plot(data[:,0],data[:,1], "ko")
                    xplt=np.linspace(data[:, 0][0], data[:, 0][-1], 1000)
                    # ax[0].plot(xplt,bckg + distrm3(xplt,*P0m), "b--")
                    dev= distrm3(data[:,0],*P0m) + distrp3(data[:,0],*P0p) - data[:,1] - gauss(data[:,0], P0m[0], -P0m[3], P0m[6])
                    if a==0:
                        dd=np.abs(dev)
                        mm=data[:,1]
                        a=1
                    else:
                        dd=np.append(dd,np.abs(dev))
                        mm=np.append(mm,data[:,1])
    print(np.average(dd), np.amax(dd))
    plt.plot(np.divide(dd+1,mm+1))
                    # ax[0].plot(xplt,bckg + distrp3(xplt,*P0p), "b--")
                    # ax[1].plot(data[:,0], dev)
                    # color=["r-","g-","k-"]
                    # for i in range(3):
                    #     ax[0].plot(xplt,(bckg+gauss(xplt, P0m[i], -P0m[i+3], P0m[i+6])), color[i%3])
                    #     ax[0].plot(xplt, (bckg+gauss(xplt, P0p[i], P0p[i+3], P0p[i+6])), color[i%3])
                # else:
                    # if(P0m[0]>0):
                          # ax[0].plot(data[:,0],data[:,1], "k-")
                          # xplt=np.linspace(data[:, 0][0], data[:, 0][-1], 1000)
                          # ax[0].plot(xplt,(bckg+gauss(xplt, P0m[0], -P0m[3], P0m[6])), color[i%3]) 
                # plt.close(fig)