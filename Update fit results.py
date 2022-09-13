#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:12:29 2022

@author: exp-k03
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

names=["bcr_1_2","bcr_1_2_3","bcr_1_2_no_div","bcr_1_2_3_no_div","bcr_1_2_phi","bcr_1_2_3_phi_1_2", "bcr_1_2_3_4_phi_1_2_3"]

krange=range(len(foldername))
# for name in names:
#     tot_fit_res = np.loadtxt(sorted_fold_path+"Total results/tot_fit_results_"+name+".mpa",skiprows=1)
#     tot_fit_cov =  np.loadtxt(sorted_fold_path+"Total results/tot_fit_covariances_"+name+".mpa",skiprows=1)
#     for k in krange:
#         data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
#         p=tot_fit_res[k,1:]
#         # print(p)
#         cov=tot_fit_cov[k,1:]
#         # print(cov)
#         with open(data_analysis+foldername[k]+"_fit_results_"+name+".mpa", "w") as f:
#             np.savetxt(f,(p,cov), header="mu sigma tau x_0 zeta_0", fmt="%.6f")

measur_groups=[[0,2,3,4,5,6],[6,7,8,9,10,11,12],[1], range(13)]

for group in [0,1,2,3]: #0 for Juergen, 1 for Martin, 2 for Christian, 3 for all
    krange=measur_groups[group]
    tot_fit_res = np.loadtxt(sorted_fold_path+"Total results 13 sep/group_"+str(group)+"_multi_fit_results_"+names[4]+".mpa",skiprows=1, dtype=float)
    print(tot_fit_res)
    c = tot_fit_res**0.5
    p=tot_fit_res
    L=(len(p)-3)//len(krange)
    # print(L)
    kaus=0
    for k in krange:
        data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
        P=[*p[0:3],*p[3+kaus*L:3+L*(1+kaus)]]
        # c=np.diag(cov)**0.5
        COV=[*c[0:3],*c[3+kaus*L:3+L*(1+kaus)]]
        # print(P)
        with open(data_analysis+foldername[k]+"_group_"+str(group)+"_multi_fit_PAR_"+names[4]+".mpa", "w") as f:
            np.savetxt(f,(P,COV), header="PAR")
        kaus+=1