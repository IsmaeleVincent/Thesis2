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

krange=[1]#range(0,len(foldername))

"""
This block calculates the diffraction efficiencies, first part estimates theta=0
and second part the diff eff for each theta
"""
plot=0
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
    for z in range(len(stack[0,0,:])):
        zprofile0 = np.zeros(len(stack[0,0,:]))
        zprofile0 += stack[yabsmax,xabsmax,:]
        # for i in range(3):
        #     for j in range(3):
        #         zprofile0 += stack[yabsmax+i-1,xabsmax+j-1,:].copy()/6
    zmin1=roi[:,7][roi[:,0]==yabsmax]
    zmin2=roi[:,8][roi[:,0]==yabsmax]
    f2 = interp1d(np.where(zprofile0)[0], zprofile0, kind='cubic')
    zplt=np.linspace(0,len(zprofile0)-1, 10000)
    # if (k==6):
    #     zplt=np.linspace(4,len(zprofile0)-1, 10000)
    zplt1=np.linspace(zmin1,zmin2, 1000)
    zmax= zplt1[f2(zplt1)==np.amax(f2(zplt1))]
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
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.set_title(foldername[k])
        ax.axvline(zmax, color="b")
        ax.plot(np.where(zprofile0)[0],zprofile0, "ko")
        ax.plot(zplt,f2(zplt), "b-")
        ax.axvline(z1, color="r")
        ax.axvline(z2, color="g")
        ax.axvline(c, color="k")
    P0m = np.zeros(9)
    P0p = np.zeros(9)
    #print(foldername[k])
    for y in range(len(roi[:,0])):
        for z in range(len(stack[0,0,:])):
            diff_eff[z][0]=(z-c)*step_theta[k]
            bckg = 0#data_and_fit[z*len(roi[:,0])+y][2]
            for j in range(len(P0m)):
                P0m[j]=data_and_fit[z*len(roi[:,0])+y][j+3] 
                P0p[j]=data_and_fit[z*len(roi[:,0])+y][j+3+len(P0m)]
            data=np.zeros((roi[y][2]-roi[y][1]+1,2))
            data[:,0] =  np.arange(roi[y][1],roi[y][2]+1)
            data[:,1] =  stack[roi[y][0],roi[y][1]:(roi[y][2]+1),z]
            data[:,0] = (data[:,0]-xabsmax)
            xplt=data[:, 0]
            color=["r-","g-","k-"]
            print(roi[y,0],z)
            if npeaks[y,0]==0 and npeaks[y,1]==0:
                diff_eff[z][6]+=sum(data[:,1])
                diff_eff[z][7]+=sum(data[:,1]**0.5)
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"r")
            if npeaks[y,0]==1 and npeaks[y,1]==0:
                xm=np.zeros((1,2),dtype=int) # xm are the x coordinates of the gaussian at +-3.5*sigma  
                xm[0,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][0])[0][0]
                xm[0,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][4]+=sum(data[:xm[0,0],1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][5]+=sum(data[:xm[0,0],1]**0.5)
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[:xm[0,0],0],data[:xm[0,0],1],"g")
                
            if npeaks[y,0]==0 and npeaks[y,1]==1:
                xm=np.zeros((1,2),dtype=int) # xm are the x coordinates of the gaussian at +-3.5*sigma  
                xm[0,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][0])[0][0]
                xm[0,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][8]+=sum(data[xm[0,1]:,1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][9]+=sum(data[xm[0,1]:,1]**0.5)
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[xm[0,1]:,0],data[xm[0,1]:,1],"g")
            
            if npeaks[y,0]==1 and npeaks[y,1]==1:
                xm=np.zeros((1,2),dtype=int) # xm are the x coordinates of the gaussians at +-3.5*sigma  
                xm[0,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][0])[0][0]
                xm[0,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3])<=3.5*P0m[6]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][4]+=sum(data[:xm[0,0],1])
                diff_eff[z][8]+=sum(data[xm[0,1]:,1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][5]+=sum(data[:xm[0,0],1]**0.5)
                diff_eff[z][9]+=sum(data[xm[0,1]:,1]**0.5)
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[:xm[0,0],0],data[:xm[0,0],1],"g")
                ax.plot(data[xm[0,1]:,0],data[xm[0,1]:,1],"g")
            
            if npeaks[y,0]==2 and npeaks[y,1]==1:
                xm=np.zeros((3,2),dtype=int)
                xp=np.zeros((1,2),dtype=int)
                for j in range(3):
                    xm[j,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][0])[0][0]
                    xm[j,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][-1])[0][0]+1
                    if j==1:
                        xp[0,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][0])[0][0]
                        xp[0,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][8]+=sum(data[xm[0,1]+1:,1])
                diff_eff[z][4]+=sum(data[xm[2,1]:xm[0,0],1])
                diff_eff[z][2]+=sum(data[:xm[1,0],1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][9]+=sum(data[xm[0,1]+1:,1]**0.5)
                diff_eff[z][5]+=sum(data[xm[2,1]:xm[0,0],1]**0.5)
                diff_eff[z][3]+=sum(data[:xm[1,0],1]**0.5)
                frac=np.zeros((len(data[xm[1,0]:xm[2,1],0]),2))
                if xm[1,0]<xm[2,1]:
                    for j in range(xm[1,0],xm[2,1]):
                        frac[j-xm[1,0],0]=data[j,1]*gauss(xplt[j], P0m[1], -P0m[4], P0m[7])/(gauss(xplt[j], P0m[1], -P0m[4], P0m[7])+gauss(xplt[j], P0m[2], -P0m[5], P0m[8]))
                        diff_eff[z][4]+=frac[j-xm[1,0],0]
                        diff_eff[z][5]+=frac[j-xm[1,0],0]**0.5
                        frac[j-xm[1,0],1]=data[j,1]*gauss(xplt[j], P0m[2], -P0m[5], P0m[8])/(gauss(xplt[j], P0m[1], -P0m[4], P0m[7])+gauss(xplt[j], P0m[2], -P0m[5], P0m[8]))
                        diff_eff[z][2]+=frac[j-xm[1,0],1]
                        diff_eff[z][3]+=frac[j-xm[1,0],1]**0.5
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[xm[2,1]:xm[0,0],0],data[xm[2,1]:xm[0,0],1],"g")
                ax.plot(data[xm[0,1]+1:,0],data[xm[0,1]+1:,1],"g")
                ax.plot(data[:xm[1,0],0],data[:xm[1,0],1],"k")
                if xm[1,0]<xm[2,1]:
                        ax.plot(data[xm[1,0]:xm[2,1],0],data[xm[1,0]:xm[2,1],1],"g-")
                        ax.plot(data[xm[1,0]:xm[2,1],0],data[xm[1,0]:xm[2,1],1],"k--")
                        ax.plot(data[xm[1,0]:xm[2,1],0],frac[:,0],"go")
                        ax.plot(data[xm[1,0]:xm[2,1],0],frac[:,1],"ko")
                        
            if npeaks[y,0]==1 and npeaks[y,1]==2:
                xm=np.zeros((2,2),dtype=int)
                xp=np.zeros((2,2),dtype=int)
                for j in range(3):
                    if j<2:
                        xm[j,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][0])[0][0]
                        xm[j,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][-1])[0][0]+1
                    if j>0:
                        xp[j-1,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][0])[0][0]
                        xp[j-1,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][8]+=sum(data[xm[0,1]+1:xp[1,0],1])
                diff_eff[z][10]+=sum(data[xp[0,1]+1:,1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][9]+=sum(data[xm[0,1]+1:xp[1,0],1]**0.5)
                diff_eff[z][11]+=sum(data[xp[0,1]+1:,1]**0.5)
                frac=np.zeros((len(data[xp[1,0]:xp[0,1]+1,0]),2))
                if xp[0,1]>xp[1,0]:
                    for j in range(xp[1,0],xp[0,1]):
                        frac[j-xp[1,0],0]=data[j,1]*gauss(xplt[j], P0p[1], P0p[4], P0p[7])/(gauss(xplt[j], P0p[1], P0p[4], P0p[7])+gauss(xplt[j], P0p[2], P0p[5], P0p[8]))
                        diff_eff[z][8]+=frac[j-xp[1,0],0]
                        diff_eff[z][9]+=frac[j-xp[1,0],0]**0.5
                        frac[j-xp[1,0],1]=data[j,1]*gauss(xplt[j], P0p[2], P0p[5], P0p[8])/(gauss(xplt[j], P0p[1], P0p[4], P0p[7])+gauss(xplt[j], P0p[2], P0p[5], P0p[8]))
                        diff_eff[z][10]+=frac[j-xp[1,0],1]
                        diff_eff[z][11]+=frac[j-xp[1,0],1]**0.5
                diff_eff[z][4]+=sum(data[:xm[0,0],1])
                diff_eff[z][5]+=sum(data[:xm[0,0],1]**0.5)
                
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[:xm[0,0],0],data[:xm[0,0],1],"g")
                ax.plot(data[xm[0,1]+1:xp[1,0],0],data[xm[0,1]+1:xp[1,0],1],"g")
                ax.plot(data[xp[0,1]+1,0],data[xp[0,1]+1,1],"k")
                if xp[0,1]>xp[1,0]:
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],data[xp[1,0]:xp[0,1]+1,1],"g-")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],data[xp[1,0]:xp[0,1]+1,1],"k--")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],frac[:,0],"go")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],frac[:,1],"ko")
            
            if npeaks[y,0]==2 and npeaks[y,1]==2:
                xm=np.zeros((3,2),dtype=int)
                xp=np.zeros((2,2),dtype=int)
                for j in range(3):
                    xm[j,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][0])[0][0]
                    xm[j,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]+P0m[3+j])<=3.5*P0m[6+j]][-1])[0][0]+1
                    if j>0:
                        xp[j-1,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][0])[0][0]
                        xp[j-1,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]-P0p[3+j])<=3.5*P0p[6+j]][-1])[0][0]+1
                diff_eff[z][6]+=sum(data[xm[0,0]:xm[0,1],1])
                diff_eff[z][4]+=sum(data[xm[2,1]:xm[0,0],1])
                diff_eff[z][2]+=sum(data[:xm[1,0],1])
                diff_eff[z][7]+=sum(data[xm[0,0]:xm[0,1],1]**0.5)
                diff_eff[z][5]+=sum(data[xm[2,1]:xm[0,0],1]**0.5)
                diff_eff[z][3]+=sum(data[:xm[1,0],1]**0.5)
                fracm=np.zeros((len(data[xm[1,0]:xm[2,1],0]),2))
                if xm[1,0]<xm[2,1]:
                    for j in range(xm[1,0],xm[2,1]):
                        fracm[j-xm[1,0],0]=data[j,1]*gauss(xplt[j], P0m[1], -P0m[4], P0m[7])/(gauss(xplt[j], P0m[1], -P0m[4], P0m[7])+gauss(xplt[j], P0m[2], -P0m[5], P0m[8]))
                        diff_eff[z][4]+=fracm[j-xm[1,0],0]
                        diff_eff[z][5]+=fracm[j-xm[1,0],0]**0.5
                        fracm[j-xm[1,0],1]=data[j,1]*gauss(xplt[j], P0m[2], -P0m[5], P0m[8])/(gauss(xplt[j], P0m[1], -P0m[4], P0m[7])+gauss(xplt[j], P0m[2], -P0m[5], P0m[8]))
                        diff_eff[z][2]+=fracm[j-xm[1,0],1]
                        diff_eff[z][3]+=fracm[j-xm[1,0],1]**0.5
                diff_eff[z][8]+=sum(data[xm[0,1]+1:xp[1,0],1])
                diff_eff[z][10]+=sum(data[xp[0,1]+1:,1])
                diff_eff[z][9]+=sum(data[xm[0,1]+1:xp[1,0],1])**0.5
                diff_eff[z][11]+=sum(data[xp[0,1]+1:,1])**0.5
                fracp=np.zeros((len(data[xp[1,0]:xp[0,1]+1,0]),2))
                if xp[0,1]>xp[1,0]:
                    for j in range(xp[1,0],xp[0,1]):
                        fracp[j-xp[1,0],0]=data[j,1]*gauss(xplt[j], P0p[1], P0p[4], P0p[7])/(gauss(xplt[j], P0p[1], P0p[4], P0p[7])+gauss(xplt[j], P0p[2], P0p[5], P0p[8]))
                        diff_eff[z][8]+=fracp[j-xp[1,0],0]
                        diff_eff[z][9]+=fracp[j-xp[1,0],0]**0.5
                        fracp[j-xp[1,0],1]=data[j,1]*gauss(xplt[j], P0p[2], P0p[5], P0p[8])/(gauss(xplt[j], P0p[1], P0p[4], P0p[7])+gauss(xplt[j], P0p[2], P0p[5], P0p[8]))
                        diff_eff[z][10]+=fracp[j-xp[1,0],1]
                        diff_eff[z][11]+=fracp[j-xp[1,0],1]**0.5
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot()
                ax.plot(data[:,0],data[:,1],"-")
                ax.plot(data[xm[0,0]:xm[0,1],0],data[xm[0,0]:xm[0,1],1],"r")
                ax.plot(data[xm[2,1]:xm[0,0],0],data[xm[2,1]:xm[0,0],1],"g")
                ax.plot(data[xm[0,1]+1:xp[1,0],0],data[xm[0,1]+1:xp[1,0],1],"g")
                ax.plot(data[:xm[1,0],0],data[:xm[1,0],1],"k")
                if xm[1,0]<xm[2,1]:
                        ax.plot(data[xm[1,0]:xm[2,1],0],data[xm[1,0]:xm[2,1],1],"g-")
                        ax.plot(data[xm[1,0]:xm[2,1],0],data[xm[1,0]:xm[2,1],1],"k--")
                        ax.plot(data[xm[1,0]:xm[2,1],0],fracm[:,0],"go")
                        ax.plot(data[xm[1,0]:xm[2,1],0],fracm[:,1],"ko")
                if xp[0,1]>xp[1,0]:
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],data[xp[1,0]:xp[0,1]+1,1],"g-")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],data[xp[1,0]:xp[0,1]+1,1],"k--")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],fracp[:,0],"go")
                        ax.plot(data[xp[1,0]:xp[0,1]+1,0],fracp[:,1],"ko")
                
            ax.set_title(foldername[k] +'-Line ' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z))+ "\nnum peaks ="+str(npeaks[y][0])+", "+str(npeaks[y][1]))
            plt.savefig(controldiff+foldername[k] +'_line_' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z))+'_fit.png')
            plt.close(fig)
            if(k==6 and (z==3 or z==4)):
                diff_eff[z][:]*=0
    # diff_eff[:,3::2]=diff_eff[:,2::2]**0.5
    with open(data_analysis+foldername[k]+'_diff_eff_new.mpa', 'w') as f:
        np.savetxt(f,diff_eff, header="theta err counts-2 err counts-1 err counts-0 err counts1 err counts1 err", fmt="%.6f")


"""
# This block plots the diffraction efficiencies
"""

# for k in krange:#range(6,len(foldername)):#
#     data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
#     diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)
#     fig = plt.figure(figsize=(15,15))
#     ax = fig.add_subplot(111)
#     ax.set_title(foldername[k])
#     for j in range(5):
#         ax.plot(diff_eff[:,0],diff_eff[:,2*j+2],'o')
#         ax.errorbar(diff_eff[:,0],diff_eff[:,2*j+2],yerr=diff_eff[:,2*j+3],capsize=1)

"""
# This block copies the plots in a common folder
"""

# if os.path.exists(allcontroldiff):
#     shutil.rmtree(allcontroldiff)
# os.makedirs(allcontroldiff)
# for k in range(0,1):#len(foldername)):
#     data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
#     controldiff = data_analysis + "Control Diff/" 
#     folder = sorted_fold_path+foldername[k]
#     roi =  np.loadtxt(data_analysis+foldername[k]+'_ROI+Peaks.mpa',skiprows=1).astype(int)
#     for y in range(len(roi[:,0])):
#         for z in range(n_theta[k]):
#             contdiffname = controldiff+foldername[k] +'_line_' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z))+'_fit.png'
#             try:
#                 shutil.copy(controldiff+contdiffname, allcontroldiff+contdiffname)   
#                 print("here")
#             except FileNotFoundError:
#                 a=0
#                 print("not there")