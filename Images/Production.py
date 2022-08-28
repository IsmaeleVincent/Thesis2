#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 23:13:11 2022

@author: aaa
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.fontsize"] = 15
fig = plt.figure(figsize=(10,5))#constrained_layout=True
gs = GridSpec(8, 1, figure=fig,hspace=0, top=0.95)
ax = [fig.add_subplot(gs[0:3, 0]),
      fig.add_subplot(gs[7, 0]),
      fig.add_subplot(gs[4:7, 0])]

for i in [0,1,2]:
    ax[i].tick_params(axis="both", labelbottom=False, bottom = False, labelleft=False, left= False)
    ax[i].set_xlim([0,1])
    ax[i].set_xlim([0,1])
n=3
n_loop=90
size=13
T=np.pi*3

for i in range(n_loop-10):
    pol=np.random.rand(2,n)
    nano=np.random.rand(2,n)
    if i==0:
        ax[0].plot(pol[0],pol[1], "o", color=(0.5,0.5,0.5), ms=size, mec="k", mew=1, label="Nanoparticles")
        ax[0].plot(nano[0],nano[1], "wo", ms=size, mec="k", label="Monomers")
    else:
        ax[0].plot(pol[0],pol[1], "o", color=(0.5,0.5,0.5), ms=size, mec="k", mew=1)
        ax[0].plot(nano[0],nano[1], "wo", ms=size, mec="k")
ax[0].set_title("Uniform mixture", fontsize=20)
ax[2].set_title("Sinusoidal modulation", fontsize=20)
ax[0].legend(loc=1)
matrix=np.zeros((1000,1000))
x=np.linspace(0, 1,1000)
for i in range(1000):
    matrix[i]=np.cos(T*2*(x+0*np.pi))
ax[1].imshow(matrix,cmap='Greys')
ax[1].set_ylim([0,65])
ax[1].set_xlim([0,1000])
# ax[1].plot(x,np.cos(T*2*(x+np.pi/4)), "k--", lw=3, label="Light intensity")
# ax[1].legend(loc=4)
ax[1].set_xlabel("$Light \ intensity$",fontsize=15)
x0=np.arange(0,1//(np.pi/T)+2)*np.pi/T
for j in range(n_loop):
    pol=np.random.rand(2,n)
    nano=np.random.rand(2,n)
    x=np.linspace(x0[j%len(x0)],x0[j%len(x0)]+np.pi/T, 10000)
    a=np.cos(T*x)**4
    spl = UnivariateSpline(x, a, k=4, s=0)
    I=spl.antiderivative()(x)
    y=np.random.rand(n)*np.amax(I)
    xp=np.zeros(n)
    for i in range(n):
        aus =abs(I-y[i])
        xp[i]=x[aus==np.amin(aus)]
    if j==14:
        ax[2].plot(xp,pol[0], "o", color=(0.5,0.5,0.5), ms=size, mec="k", mew=1)
    else:
        ax[2].plot(xp,pol[0], "o", color=(0.5,0.5,0.5), ms=size, mec="k", mew=1)
    a=np.sin(T*x)**4
    spl = UnivariateSpline(x, a, k=4, s=0)
    I=spl.antiderivative()(x)
    y=np.random.rand(n)*np.amax(I)
    xp=np.zeros(n)
    for i in range(n):
        aus =abs(I-y[i])
        xp[i]=x[aus==np.amin(aus)]
    a=j%3
    if (a==0 or a==1):
        ax[2].plot(xp,nano[1], "-", color=(0.,0.2,0.3),ms=size, mec="k", lw=4)
        ax[2].plot(xp,nano[1], "wo",ms=size, mec="k")

    else:
        # ax[2].plot(xp,nano[1], "-", color=(0.,0.1,0.2),ms=size, mec="k", lw=3)
        ax[2].plot(xp,nano[1], "wo", ms=size, mec="k")
# fig.legend(ncol=3, bbox_to_anchor=(0.6, 0.47, 0.1, 0.1))
plt.savefig('Production.eps', format='pdf',bbox_inches='tight')