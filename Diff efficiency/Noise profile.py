#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:44:36 2022

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
n_pixel = 16384 #number of pixels in one measurement

def ps1(k, mu):
    return ps.pmf(k,mu)
def normal(x, mu,s):
    return norm.pdf(x,mu,s)

for k in range(len(foldername)):
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    matrixes = [np.loadtxt(sorted_fold_path+foldername[k]+"/Matrixes/"+foldername[k]+"_"+str("%03d" % (j,))+".mpa") for j in range (1,n_theta[k]+1)]
    stack = np.stack(matrixes,axis=2)
    xyzabsmax = np.where(stack[:,:,:]==np.amax(stack[:,:,:]))
    yabsmax = xyzabsmax[0][0]
    xabsmax = xyzabsmax[1][0]
    zabsmax = xyzabsmax[2][0]
    roi =  np.loadtxt(data_analysis+foldername[k]+'_ROI+Peaks.mpa',skiprows=1).astype(int)
    print(foldername[k])
    data_and_fit = np.zeros((len(stack[0,0,:])*len(roi[:,0]), 21))
    xmin= int(np.amin(roi[:,1]))
    xmax= int(np.amax(roi[:,2]))
   # noise=np.zeros(n_theta[k]-1)
   # for z in range(n_theta[k]-1):
        # for y in range(6):
        #     noise[z]+= np.sum(stack[yabsmax-2+i, roi[y,1]:roi[y,2],z])
    noise= np.ravel(stack[0:10, :,:])
    #noise = np.ravel(stack[yabsmax-20:yabsmax-10,xmin:xmax,:])
    # print(np.amax(noise))
    # print(noise)
    # IQR  = stats.iqr(noise, nan_policy="omit")
    # N    = noise.size
    # bw   = (2 * IQR) / np.power(N, 1/3)
    # b= int((np.amax(noise)-np.amin(noise)) / bw+1)
    b=np.arange(int(np.amax(noise))+2)-0.5
    entries, bin_edges= np.histogram(noise,bins=b,density="true")
    # print(entries, bin_edges)
    bin_mid = (0.5 * (bin_edges[1:] + bin_edges[:-1]))
    x= np.around(bin_mid)
    # print(x)
    # #delta=(bin_edges[-1]-bin_edges[0])/b
    # print(bin_mid)
    p, cov = fit(ps1, x, entries, p0=np.sum(noise)/len(noise))#bin_mid[entries==np.amax(entries)][0])
    chi=chisquare(entries,ps.pmf(x,*p))
    #print(chi)
    # print(cov,p)
    p1, cov1 = fit(normal, x, entries, p0=[np.sum(noise)/len(noise),100])
    chi1=chisquare(entries,norm.pdf(x,*p1))
    W=bin_mid[1]-bin_mid[0]
    plt.hist(noise,bins=b,label="Counts")#,histtype='stepfilled')
    plt.bar(x, ps.pmf(x,*p)/np.amax(entries)*np.amax(np.histogram(noise,bins=b)[0]), color="red", width=W/2, alpha=0.5, label="\nPoisson fit \n"+"$\mu$="+str("%.3f"%p[0])+"\n$\sqrt{\mu}$="+str("%.3f"%p[0]**0.5) + "\n$p value$="+str("%.3f"%chi[1]))#width=delta/2,
    #plt.bar(x, norm.pdf(x,*p1)/np.amax(entries)*np.amax(np.histogram(noise,bins=b)[0]), color="green", width=-W/2,align="edge", alpha=0.5, label="\nGaussian fit \n"+"$\mu$="+str("%.3f"%p1[0])+"\n$\sigma$="+str("%.3f"%p1[1])+ "\n$p value$="+str("%.3f"%chi1[1]))#width=delta/2,
    #plt.xlim([np.amin(noise), np.amax(noise)])
    plt.xlim([np.amin(x)-W/2, np.amax(x)+W])
    plt.title(foldername[k])
    plt.legend()
    plt.show()
    # fig = plt.figure(figsize=(15,15))
    # ax = fig.add_subplot(211)
    # ax1 = fig.add_subplot(212)
    # matrix = np.sum(stack[yabsmax-2:yabsmax+4, xmin:xmax,:],axis=2)
    # im=ax.imshow(matrix,cmap='plasma')
    
    # ax1.plot(((noise-np.sum(noise)/len(noise))**2/(len(noise)-1))**0.5)