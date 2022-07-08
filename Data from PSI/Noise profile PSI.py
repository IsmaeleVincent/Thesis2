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
import matplotlib.ticker as ticker

orig_fold_path="/home/aaa/Desktop/thesis_L1/Data from PSI/"
fold_name = "D0043"
sorted_fold_path="/home/aaa/Desktop/thesis_L1/Data from PSI/Sorted data/" #insert folder of sorted meausements files
renamed = sorted_fold_path+fold_name+"/Renamed/"
matrixes = sorted_fold_path+fold_name+"/Matrixes/"
pictures = sorted_fold_path+fold_name+"/Pictures/"
rawpictures = sorted_fold_path+fold_name+"/Raw pictures/"
th_matrixes = sorted_fold_path+fold_name+"/Theta matrixes/"
th_pictures = sorted_fold_path+fold_name+"/Theta pictures/"
th_rawpictures = sorted_fold_path+fold_name+"/Theta raw pictures/"
n_meas=150 #number of measurements files for each folder
n_pixel = 16384 #number of pixels in one measurement

def ps1(k, mu):
    return ps.pmf(k,mu)
matrixes1 = [np.loadtxt(matrixes+fold_name+"_"+str(j)+".mpa") for j in range(n_meas)]
stack = np.stack(matrixes1,axis=2)
xyzabsmax = np.where(stack[:,:,:]==np.amax(stack[:,:,:]))
yabsmax = xyzabsmax[0][0]
xabsmax = xyzabsmax[1][0]
zabsmax = xyzabsmax[2][0]
noise = np.ravel(stack[0:20,:,:])
noise = np.delete(noise, np.where(noise<3))
print(np.amax(noise))
# IQR  = stats.iqr(noise, nan_policy="omit")
# N    = noise.size
# bw   = (2 * IQR) / np.power(N, 1/3)
# b= int((np.amax(noise)-np.amin(noise)) / bw+1)
b=int(np.amax(noise))
entries, bin_edges= np.histogram(noise,bins=b,density="true")
print(entries, bin_edges)
bin_mid = (0.5 * (bin_edges[1:] + bin_edges[:-1]))
x= np.around(bin_mid)
print(x)
delta=(bin_edges[-1]-bin_edges[0])/b
print(bin_mid)
p, cov = fit(ps1, x, entries, p0=bin_mid[entries==np.amax(entries)][0])
print(cov,p)
plt.hist(noise,bins=b,label="Counts",histtype='stepfilled')
plt.bar(x, ps.pmf(x,*p)/np.amax(entries)*np.amax(np.histogram(noise,bins=b)[0]),width=delta/2, color="red", alpha=0.8, label="Poisson fit")
plt.title("D0043")
plt.legend()
plt.show()