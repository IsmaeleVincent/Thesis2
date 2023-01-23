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

n_diff= 2 #number of peaks for each side, for example: n=2 for 5 diffracted waves

pendel_data = np.zeros((len(tilt),4))
pendel_data[:,0] = tilt
krange=np.arange(len(foldername))
krange=krange[np.argsort(tilt[krange])]
for k in krange:
    data_analysis = sorted_fold_path+foldername[k]+"/Data Analysis/"
    diff_eff =  np.loadtxt(data_analysis+foldername[k]+'_diff_eff_new.mpa',skiprows=1)    
    pendel_data[k,0]=diff_eff[:,4][diff_eff[:,4]==np.amax(diff_eff[:,4])]
    pendel_data[k,1]=diff_eff[:,5][diff_eff[:,4]==np.amax(diff_eff[:,4])]
    pendel_data[k,2]=diff_eff[:,2][diff_eff[:,2]==np.amax(diff_eff[:,2])]
    pendel_data[k,3]=diff_eff[:,3][diff_eff[:,2]==np.amax(diff_eff[:,2])]
    
fig = plt.figure(figsize=(5,4))
ax=fig.add_subplot(111)
# print(tilt[krange])
ax.errorbar(tilt[krange], pendel_data[krange,0], yerr=pendel_data[krange,1], fmt="^-k", capsize=3, label= "Data order -1")#color=(0.5,0.5,0.5),
ax.errorbar(tilt[krange], pendel_data[krange,2], yerr=pendel_data[krange,3], fmt="^-", capsize=3, color=(0.8,0,0), label="Data order -2")
ax.set_xlabel("$\zeta$ (deg)")
ax.set_ylabel("Max. diffraction efficiencies")
ax.legend(loc=6)

plt.tight_layout()
plt.show()
# plt.savefig("Data_example.pdf", format='pdf', bbox_inches='tight')
