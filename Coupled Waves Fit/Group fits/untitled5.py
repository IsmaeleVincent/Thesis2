#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:01:10 2022

@author: aaa
"""

import numpy as np

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
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measurement


krange=range(len(foldername))
tot_fit_res = np.loadtxt(sorted_fold_path+'tot_fit_results_bcr3.mpa',skiprows=1)
tot_fit_cov =  np.loadtxt(sorted_fold_path+'tot_fit_covariances_bcr3.mpa',skiprows=1)
for k in krange:
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    p=tot_fit_res[k,1:]
    print(p)
    cov=tot_fit_cov[k,1:]
    print(cov)
    with open(data_analysis+foldername[k]+'_fit_results_bcr3.mpa', 'w') as f:
        np.savetxt(f,(p,cov), header="mu sigma tau x_0 zeta_0", fmt="%.6f")
