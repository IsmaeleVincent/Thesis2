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
from scipy.interpolate import UnivariateSpline

orig_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/"
fold_name = "D0043"#"NP829"
sorted_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/Sorted data/" #insert folder of sorted meausements files
renamed = sorted_fold_path+fold_name+"/Renamed/"
matrixes = sorted_fold_path+fold_name+"/Matrixes/"
pictures = sorted_fold_path+fold_name+"/Pictures/"
rawpictures = sorted_fold_path+fold_name+"/Raw pictures/"
th_matrixes = sorted_fold_path+fold_name+"/Theta matrixes/"
th_pictures = sorted_fold_path+fold_name+"/Theta pictures/"
th_rawpictures = sorted_fold_path+fold_name+"/Theta raw pictures/"
data_analysis = sorted_fold_path+fold_name+"/Data analysis/"
n_meas=175 #number of measurements files for each folder
n_pixel = 16384 #number of pixels in one measurement
n_runs = 7
n_theta = n_meas//n_runs
"""
This block calculates the region of interest (ROI)
"""
# matrixes1 = [np.loadtxt(th_matrixes+fold_name+"_"+str(j)+".mpa") for j in range(n_theta)]
# stack = np.stack(matrixes1,axis=2)
# sum_mx = np.sum(stack, axis=2)
# s_max = np.amax(sum_mx)
# print(np.where(sum_mx==s_max))
# noise=np.sum(sum_mx[:,0:30]/sum_mx[:,0:30].size)
# max_pos=np.zeros((5,2),dtype=int)
# max_pos[2]=np.where(sum_mx==s_max)

# for i in range(2):
#     max_pos[1-i,0] = np.where(sum_mx[30:60,30:max_pos[2-i,1]-4]==np.amax(sum_mx[30:60,:max_pos[2-i,1]-4]))[0][0]
#     max_pos[1-i,1] = np.where(sum_mx[30:60,30:max_pos[2-i,1]-4]==np.amax(sum_mx[30:60,:max_pos[2-i,1]-4]))[1][0]
#     max_pos[1-i,0]+=30
#     max_pos[1-i,1]+=30
#     max_pos[3+i,0] = np.where(sum_mx[30:60,max_pos[2+i,1]+4:]==np.amax(sum_mx[30:60,max_pos[2+i,1]+4:100]))[0][0]
#     max_pos[3+i,1] = np.where(sum_mx[30:60,max_pos[2+i,1]+4:]==np.amax(sum_mx[30:60,max_pos[2+i,1]+4:100]))[1][0]
#     max_pos[3+i,0]+=30
#     max_pos[3+i,1]+=max_pos[2+i,1]+4
# for i in range(128):
#     for j in range(128):
#         if(j==0 or i==0 or j==127 or i==127):
#             sum_mx[i][j]=0
#         if((j>0 and i>0) and (j<127 and i<127)):
#             if (sum_mx[i+1][j+1]+sum_mx[i+1][j-1]+sum_mx[i-1][j-1]+sum_mx[i+1][j-1]<1):
#                 sum_mx[i][j]=0
# lines=np.zeros(2,dtype=int)
# j=0
# a=1
# while a>0:
#     a=sum_mx[max_pos[2][0]-j][max_pos[2][1]]
#     j+=1
# print(max_pos[2][0]-j)
# lines[0]=max_pos[2][0]-j
# j=0
# a=1
# while a>0:
#     a=sum_mx[max_pos[2][0]+j][max_pos[2][1]]
#     j+=1
# print(max_pos[2][0]+j)
# lines[1]=max_pos[2][0]+j
# print(max_pos)
# rows=np.zeros((5,2),dtype=int)
# for i in range(5):
#     j=0
#     a=1
#     while a>0:
#         a=np.amax(sum_mx[lines[0]:lines[1],max_pos[i][1]-j])
#         j+=1
#     rows[i][0]=max_pos[i][1]-j
#     j=0
#     a=1
#     while a>0:
#         a=np.amax(sum_mx[lines[0]:lines[1],max_pos[i][1]+j])
#         j+=1
#     rows[i][1]=max_pos[i][1]+j
# sum_mx[sum_mx>s_max/150] = s_max/150
# for i in range(5):
#     sum_mx[max_pos[i,0],max_pos[i,1]]=s_max/50
#     sum_mx[:,rows[i,0]]=s_max/150
#     sum_mx[:,rows[i,1]]=s_max/150
# plt.imshow(sum_mx,cmap='plasma')
# sum_mx[lines[0],:]=s_max/150
# sum_mx[lines[1],:]=s_max/150
# a=im.fromarray(sum_mx)
# plt.imshow(sum_mx,cmap='plasma')
# print(rows)
# with open(data_analysis+fold_name+'_ROI.mpa',"w") as f:
#     np.savetxt(f,lines, header="line1 line2", fmt="%1.0f")
#     np.savetxt(f,rows, header="row1 row2 (-2,-1,0,1,2) ", fmt="%1.0f")

"""
This block calculates diffraction efficiencies
"""
matrixes1 = [np.loadtxt(th_matrixes+fold_name+"_"+str(j)+".mpa") for j in range(n_theta)]
err= [np.loadtxt(th_matrixes+fold_name+"_"+str(j)+"_err.mpa") for j in range(n_theta)]
stack = np.stack(matrixes1,axis=2)
stack_err = np.stack(err,axis=2)
lines= np.loadtxt(data_analysis+fold_name+'_ROI.mpa',dtype=int,skiprows=1,max_rows=2)
rows= np.loadtxt(data_analysis+fold_name+'_ROI.mpa',dtype=int,skiprows=3)
print(lines)
diff_eff= np.zeros((len(stack[0,0,:]),12))
diff_eff[:,0]=np.linspace(-1.1994,1.199252, len(stack[0,0,:]))
print()
diff_eff[:,1]=0.01
for i in range(len(stack[0,0,:])):
    for j in range(5):
        diff_eff[i,2*j+2]=np.sum(stack[lines[0]:lines[1],rows[j,0]:rows[j,1],i])
        diff_eff[i,2*j+3]=np.sum((stack_err[lines[0]:lines[1],rows[j,0]:rows[j,1],i]))
# err=diff_eff[:,2::2]**0.5
# diff_eff[:,3::2]=err
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.set_title(fold_name)
#ax.set_ylim([0,125])
for j in range(5):
    ax.plot(diff_eff[:,0],diff_eff[:,2*j+2],'o')
    ax.errorbar(diff_eff[:,0],diff_eff[:,2*j+2],yerr=diff_eff[:,2*j+3],capsize=1)
with open(data_analysis+fold_name+'_diff_eff.mpa',"w") as f:
    np.savetxt(f,diff_eff, header="theta err -2 err -1 err 0 err 1 err 2 err", fmt="%.4f")



