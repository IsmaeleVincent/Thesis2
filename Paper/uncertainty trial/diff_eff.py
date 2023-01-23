#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:27:14 2023

@author: aaa
"""
import numpy as np
import matplotlib.pyplot as plt


data_fold="/home/aaa/Desktop/Thesis2/Paper/uncertainty trial/" #insert folder of sorted meausements files
data_name="allInt_nd061716_zeta69_20210310.dat"

"""
# This block calculates the diffraction efficiencies
"""
diff_int0 =  np.loadtxt(data_fold+data_name)
diff_int = diff_int0[:,:-2]
diff_eff2 =  np.loadtxt(data_fold+"de.dat")
# diff_int[:,1::2][diff_int[:,1::2]==0]=0.001
diff_eff=diff_int.copy()
sum_int = np.sum(diff_int[:,1::2],axis=1)
print(diff_int[0,1::2])

for i in range(len(diff_eff[:,0])):
    diff_eff[i,1::2] = diff_int[i,1::2]/sum_int[i]
print(diff_eff[0,1])
diff_eff[:,2::2]=diff_eff[:,1::2]**2-diff_eff[:,1::2]**3
diff_eff[:,2::2]=np.divide(diff_eff[:,2::2],diff_int[:,1::2])
diff_eff[:,2::2]=diff_eff[:,2::2]**0.5
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
#ax.set_ylim([0,125])
for j in range(7):
    ax.plot(diff_eff[:,0],diff_eff[:,2*j+1],'or')
    ax.errorbar(diff_eff[:,0],diff_eff[:,2*j+1],yerr=diff_eff[:,2*j+2],capsize=1)
    # ax.plot(diff_eff2[:,0],diff_eff2[:,2*j+1],'.b')
    # ax.errorbar(diff_eff2[:,0],diff_eff2[:,2*j+1],yerr=diff_eff2[:,2*j+2],capsize=1)
    
print(np.amax(abs(diff_eff[:,2::2]-diff_eff2[:,2::2])))
plt.show()
with open(data_fold+"de_new.txt", 'w') as f:
    np.savetxt(f,diff_eff, header="theta counts-2 err counts-1 err counts-0 err counts1 err counts1 err", fmt="%.6f")
