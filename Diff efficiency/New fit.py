#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:36:54 2022

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
This block calculates the fits
"""

plot=1
def gauss(x, A, x0,sx):
    return A/sx*np.exp(-(x-x0)**2/(2*(sx)**2))
def distrm3(x,A0,A1,A2,x0,x1,x2,s0,s1,s2):
    return bg1+gauss(x,A0,-x0,s0)+gauss(x, A1,-x1,s1)+gauss(x, A2, -x2,s2)
def distrp3(x,A0,A1,A2,x0,x1,x2,s0,s1,s2):
    return bg1+gauss(x,A0,x0,s0)+gauss(x, A1, x1,s1)+gauss(x, A2, x2,s2)
def distr1(x, A, x0,sx):
    return bg1+A/sx*np.exp(-(x-x0)**2/(2*(sx)**2))

for k in range(0,1):#len(foldername)):
    bckg=0.
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
    for y in range(len(roi[:,0])):
        if (roi[y][2] == roi[y][1]):
            bckg = sum(stack[roi[y][0],xabsmax-20:xabsmax-3,0])/len(stack[roi[y][0],xabsmax-20:xabsmax-3,0])
        if ((roi[y][3]==0 or roi[y][4]==0) and roi[y][2]-roi[y][1]>5):
            P0 = [1.,0.,0.5]
            bound = [[0.2,-3.,0.01,],[1.5,3.,1.]]
            for z in range(len(stack[0,0,:])):
                data=np.zeros((roi[y][2]-roi[y][1]+1,2))
                data[:,0] =  np.arange(roi[y][1],roi[y][2]+1)
                data[:,1] =  stack[roi[y][0],roi[y][1]:(roi[y][2]+1),z]
                ymax = np.amax(data[:,1])
                ymin = np.amin(data[:,1])
                bg1 = bckg/ymax
                data[:,0] = (data[:,0]-xabsmax)
                data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
                p,cov=fit(distr1,data[:,0][data[:,0]>-3],data[:,1][data[:,0]>-3], p0=P0, bounds = bound)
                P0=p.copy()
                if plot:
                    fig = plt.figure(figsize=(15,15))
                    ax = fig.add_subplot(111)
                    ax.set_title(foldername[k] +'-Line ' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z)))
                    ax.plot(data[:,0],data[:,1], "k-")
                    xplt=np.linspace(data[:, 0][0], data[:, 0][-1], 1000)
                    ax.plot(xplt, distr1(xplt,*p), "r-")
                data_and_fit[z*len(roi[:,0])+y][0] = z
                data_and_fit[z*len(roi[:,0])+y][1] = roi[y][0]
                data_and_fit[z*len(roi[:,0])+y][2] = bckg + ymin
                for j in range(len(P0)):
                    data_and_fit[z*len(roi[:,0])+y][j*3+3] = P0[j]
                data_and_fit[z*len(roi[:,0])+y][3]*=(ymax-ymin)
        if (roi[y][3]>1 or roi[y][4]>1):
            if(roi[y][3]==1):
                P0m = [1.,0.3,0.1, 0., abs(roi[y][5]-xabsmax)-2, abs(roi[y][5]-xabsmax)+8, 0.5, 0.5, 0.5]
                boundm = [[0.2,0.,0.,-2.,abs(roi[y][5]-xabsmax)-4, abs(roi[y][5]-xabsmax)+2,0.01,0.1,0.1],[1.5,1.5,0.4,2.,abs(roi[y][1]-xabsmax),abs(roi[y][1]-xabsmax), 1.,1.5,1.5]]
            if(roi[y][3]>1):    
                P0m = [1.,0.3,0.1, 0., abs(roi[y][5]-xabsmax)-2, abs(roi[y][5]-xabsmax)+8, 0.5, 0.5, 0.5]
                boundm = [[0.2,0.,0.,-2.,abs(roi[y][5]-xabsmax)-4, abs(roi[y][5]-xabsmax)+5,0.01,0.1,0.1],[1.5,1.5,0.4,2.,abs(roi[y][1]-xabsmax),abs(roi[y][1]-xabsmax), 1.,1.5,1.5]]
            if(roi[y][4]==1):
                P0p = [1.,0.,0., 0., abs(roi[y][6]-xabsmax)-2, abs(roi[y][6]-xabsmax)+3, 0.5, 1., 1.5]
                boundp = [[0.2,0,0,-2.,abs(roi[y][6]-xabsmax)-4, abs(roi[y][6]-xabsmax)+1,0.01,0.1,0.1],[2.,0.4,1e-8,2.,abs(roi[y][2]-xabsmax),abs(roi[y][2]-xabsmax), 1,1.5,1.5]]
            if(roi[y][4]>1):    
                P0p = [1.,0.,0., 0., abs(roi[y][6]-xabsmax)-2, abs(roi[y][6]-xabsmax)+3, 0.5, 1., 1.5]
                boundp = [[0.2,0,0,-2.,abs(roi[y][6]-xabsmax)-4, abs(roi[y][6]-xabsmax)+2,0.01,0.1,0.1],[2.,0.4,1e-8,2.,abs(roi[y][2]-xabsmax),abs(roi[y][2]-xabsmax), 1,1.5,1.5]]
            P0maus=P0m.copy()
            P0paus=P0p.copy()
            boundmaus=boundm.copy()
            boundpaus=boundp.copy()
            zmin1=roi[y][7]
            zmin2=roi[y][8]
            for z in range(len(stack[0,0,:])):
                data=np.zeros((roi[y][2]-roi[y][1]+1,2))
                data[:,0] =  np.arange(roi[y][1],roi[y][2]+1)
                data[:,1] =  stack[roi[y][0],roi[y][1]:(roi[y][2]+1),z]
                ymax = np.amax(data[:,1])
                ymin = np.amin(data[:,1])
                bg1 = bckg/ymax
                data[:,0] = (data[:,0]-xabsmax)
                data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
                if plot:
                    fig = plt.figure(figsize=(15,15))
                    ax = fig.add_subplot(111)
                    ax.set_title(foldername[k] +'-Line ' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z)))
                    ax.plot(data[:,0],data[:,1], "ko")
                if(roi[y][3]==0):
                    boundm[1][1] = 1e-8
                    P0m[1]=0.
                    boundm[1][2] = 1e-8
                    P0m[2]=0.
                if(z>zmin1-2):
                    if(z>zmin1+2):
                        boundm[1][2] = 1e-8
                        P0m[2]=0.
                        boundp[1][1]= 1.5
                    else:
                        boundm[1][2] = 0.1
                        boundm[1][8] = 1.
                        P0m[2]=0.09
                if(roi[y][4]==0):
                    boundp[1][1] = 1e-8
                    P0p[1]=0.
                    boundp[1][2] = 1e-8
                    P0p[2]=0.
                if(z>zmin2-4):
                    if(z>zmin2+2):
                        boundp[1][2] = 0.3
                        boundp[1][8] = 1.5
                    else:
                        boundp[1][2] = 0.1
                        boundp[1][8] = 1.
                for j in range(len(P0p)):
                    if (P0m[j]<=boundm[0][j] or P0m[j]>=boundm[1][j]):
                        P0m[j]=(boundm[1][j]-boundm[0][j])/3+boundm[0][j]
                    if (P0p[j]<=boundp[0][j] or P0p[j]>=boundp[1][j]):
                        P0p[j]=(boundp[1][j]-boundp[0][j])/3+boundp[0][j]
                    # if (boundm[0][j]>=boundm[1][j]):
                    #     print(j,"m",boundm[0][j],boundm[1][j])
                    #     boundm[0][j]=boundmaus[0][j]
                    #     boundm[1][j]=boundmaus[1][j]
                    # if (boundp[0][j]>=boundp[1][j]):
                    #     print(j,"p",boundp[0][j],boundp[1][j])
                    #     boundp[0][j]=boundpaus[0][j]
                    #     boundp[1][j]=boundpaus[1][j]
                #print(P0m, P0p)
                # for j in range(len(P0p)):
                #     if (P0p[j]<boundp[0][j] or P0p[j]>boundp[1][j]):
                #         print(P0p[j],"p",boundp[0][j],boundp[1][j])
                #     if (P0m[j]<boundm[0][j] or P0m[j]>boundm[1][j]):
                #         print(j,P0m[j],"m",boundm[0][j],boundm[1][j])
                boundmt = tuple(boundm.copy())
                boundpt = tuple(boundp.copy())
                try:
                    pm,covm=fit(distrm3,data[:,0][data[:,0]<3],data[:,1][data[:,0]<3], p0=P0m, bounds = boundmt)
                    pp,covp=fit(distrp3,data[:,0][data[:,0]>-3],data[:,1][data[:,0]>-3], p0=P0p, bounds = boundpt)
                except RuntimeError or ValueError:
                    try:
                        P0m=P0maus.copy()
                        P0p=P0paus.copy()
                        boundm=boundmaus.copy()
                        boundp=boundpaus.copy()
                        if(roi[y][3]==0):
                            boundm[1][1] = 1e-8
                            P0m[1]=0.
                            boundm[1][2] = 1e-8
                            P0m[2]=0.
                        if(z>zmin1-2):
                            if(z>zmin1+2):
                                boundm[1][2] = 1e-8
                                P0m[2]=0.
                                boundp[1][1]= 1.5
                            else:
                                boundm[1][2] = 0.1
                                boundm[1][8] = 1.
                                P0m[2]=0.09
                        if(roi[y][4]==0):
                            boundp[1][1] = 1e-8
                            P0p[1]=0.
                            boundp[1][2] = 1e-8
                            P0p[2]=0.
                        if(z>zmin2-4):
                            if(z>zmin2+3):
                                boundp[1][2] = 0.3
                                boundp[1][8] = 1.5
                            else:
                                boundp[1][2] = 0.1
                                boundp[1][8] = 1.
                        for j in range(len(P0p)):
                            if (P0m[j]<=boundm[0][j] or P0m[j]>=boundm[1][j]):
                                P0m[j]=(boundm[1][j]-boundm[0][j])/3+boundm[0][j]
                            if (P0p[j]<=boundp[0][j] or P0p[j]>=boundp[1][j]):
                                P0p[j]=(boundp[1][j]-boundp[0][j])/3+boundp[0][j]
                        boundmt = tuple(boundm.copy())
                        boundpt = tuple(boundp.copy())
                        pm,covm=fit(distrm3,data[:,0][data[:,0]<3],data[:,1][data[:,0]<3], p0=P0m, bounds = boundmt)
                        pp,covp=fit(distrp3,data[:,0][data[:,0]>-3],data[:,1][data[:,0]>-3], p0=P0p, bounds = boundpt)
                    except RuntimeError or ValueError:
                        print('error'+foldername[k] +' Line ' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z)))
                        print(P0m, P0p)
                        print(boundmt)
                        print(boundpt)
                data_and_fit[z*len(roi[:,0])+y][0] = z
                data_and_fit[z*len(roi[:,0])+y][1] = roi[y][0]
                data_and_fit[z*len(roi[:,0])+y][2] = bckg + ymin
                P0m=pm.copy()
                P0p=pp.copy()
                for j in range(len(P0m)):
                    data_and_fit[z*len(roi[:,0])+y][j+3] = P0m[j]
                    data_and_fit[z*len(roi[:,0])+y][j+3+len(P0m)] = P0p[j]
                for j in range(3):
                    data_and_fit[z*len(roi[:,0])+y][j+3]*=(ymax-ymin)
                    data_and_fit[z*len(roi[:,0])+y][j+3+len(P0m)]*=(ymax-ymin)
                if plot:
                    xplt=np.linspace(data[:, 0][0], data[:, 0][-1], 1000)
                    ax.plot(xplt, distrm3(xplt,*pm), "b--")
                    ax.plot(xplt, distrp3(xplt,*pp), "b--")
                    color=["r-","g-","k-"]
                    for i in range(3):
                        ax.plot(xplt,(bg1+gauss(xplt, pm[i], -pm[i+3], pm[i+6])), color[i%3])
                        ax.plot(xplt, (bg1+gauss(xplt, pp[i], pp[i+3], pp[i+6])), color[i%3])
                if (P0m[4]+4<abs(roi[y][1]-xabsmax)):
                    boundm[0][5]=P0m[4]+4
                else:
                    boundm[1][5]=P0m[4]+6
                    boundm[0][5]=P0m[4]+4
                boundm[0][7]=P0m[6]
                boundm[0][8]=P0m[6]
                boundp[0][8]=P0p[6]
                boundp[0][7]=P0p[6]
                if (P0p[4]+4<abs(roi[y][2]-xabsmax)):
                    boundp[0][5]=P0p[4]+4
                else:
                    boundp[1][5]=P0p[4]+6
                    boundp[0][5]=P0p[4]+4
    # with open(data_analysis+foldername[k]+'_fit+data.mpa', 'w') as f:
    #     np.savetxt(f,data_and_fit, header="theta line bckg A0m A1m A2m x0m x1m x2m s0m s1m s2m A0p A1p A2p x0p x1p x2p s0p s1p s2p", fmt="%i %i "+"%.18e "*19)