 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:09:28 2022

@author: aaa
"""
import numpy as np
import matplotlib.pyplot as plt
pi=np.pi
rad=pi/180



sorted_fold_path="/home/aaa/Desktop/Thesis2/Sorted data/" #insert folder of sorted meausements files
tiltangles=[0,40,48,61,69,71,79,80,81]
foldername=[]
for i in range(len(tiltangles)):
    foldername.append(str(tiltangles[i])+"deg")
foldername.append("79-4U_77c88deg")
foldername.append("79-8U_76c76deg")
foldername.append("79-12U_75c64deg")
foldername.append("79-16U_74c52deg")
tilt=np.array([0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52],dtype=float)
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measurement

n_diff= 2 #number of peaks for each side, for example: n=2 for 5 diffracted waves

for k in [1]:#range(len(foldername)):
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)    
    fig, ax = plt.subplots(n_diff+1,figsize=(5,4),sharex=True)
    plt.rcParams["font.size"] = 11
    fig.text(0, 0.5, '$\eta$', va='center', rotation='vertical', fontsize=11)
    fig.subplots_adjust(hspace=-1)
    ax[0].tick_params('x', bottom=False)
    ax[1].tick_params('x', bottom=False)
    # ax[0].set_title("Example of diffracted waves intensities")
    ax[0].errorbar(diff_eff[:,0]*rad,diff_eff[:,6], fmt="^-k", yerr=diff_eff[:,7],capsize=3, label="0")
    ax[0].legend(loc=7, framealpha=1)
    for i in range(1,3):
        ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6-2*i], fmt="^-k",yerr=diff_eff[:,7-2*i], capsize=3, label="-"+str(i))
        ax[i].errorbar(diff_eff[:,0]*rad,diff_eff[:,6+2*i], fmt="^-", yerr=diff_eff[:,7+2*i], capsize=3, color = (0.8,0,0),  label="+"+str(i))
        ax[i].legend(loc=5+2*i, framealpha=1)
    ax[-1].set_xlabel("$\\theta$ (rad)")
    plt.tight_layout()
    plt.show()
plt.savefig("Data_example.pdf", format='pdf', bbox_inches='tight')
