#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:52:14 2022

@author: aaa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:34:28 2022

@author: ismae
"""
from PIL import Image as im
from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import cmath
from scipy.stats import chisquare as cs

def distr(x,A,sx,x0):
    return A/(sx)*np.exp(-0.5*(x-x0)**2/sx**2)
def distr2(x,A,sx,x0):
    return A*(np.cos((x-x0)*sx))**4

matrix = np.zeros((1200,1200))

for i in range(1200):
    matrix[:,i]=np.sin(i/10)
    
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
# ax.set_xlim([550,750])
# ax.set_ylim([800,600])
im=ax.imshow(matrix,cmap='bone')
plt.axis('off')
plt.savefig("pattern.png", bbox_inches='tight')
#plt.savefig('Horiz.eps', format='eps')
# P0=[277.40655667,1.9e-1,7.90313202]
# #plt.savefig(controlfits+foldername[k] +'_line_' +str("%0d"%(roi[0][0]+y))+'_theta'+str("%0d"%(z))+'_fit.png')
# p,cov=fit(distr2,data[2:6,0],data[2:6,1],p0=P0)
# print("p=",p)
# print("cov=", np.diag(cov)**0.5)
# xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
# xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
# ax.set_xlabel("mm")
# ax.set_ylabel("Avg counts")
# ax.plot(data[:,0]-xmax,data[:,1], "go", label="Measurement")
# ax.plot(xplt-xmax, distr2(xplt,*p), "r-", label="Gaussian Fit")
# xplt1=np.linspace(0,p[1], 100)
# ax.plot(xplt1, xplt1*0)
# plt.legend(loc=1)