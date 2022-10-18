#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:26:10 2022

@author: aaa
"""

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,3), dpi=200)
ax=fig.add_subplot()
x=np.linspace(0,10, 1000)
x1=np.linspace(-0.5,0, 100)
x2=np.linspace(10,10.5, 100)
ax.plot(x, np.cos(x)+1,color=(0.1,0.1,0.33))
ax.plot(x1, np.cos(x1)+1,"--", color=(0.1,0.1,0.33))
ax.plot(x2, np.cos(x2)+1,"--" ,color=(0.1,0.1,0.33))
ax.plot(x, x*0+1, "-k", ls="dashed")
ax.plot([0,2*np.pi],[2.1,2.1], "-|k")
ax.vlines(2*np.pi, 1, 2, ls="dotted")
ax.text( -0.1, 1,"$n_0$", ha="right",va="center", fontsize=14)
ax.text( 2*np.pi+0.1, 1.5,"$\Delta n_1$", va="center", fontsize=14)
ax.text( np.pi, 2.2,"$\Lambda$", ha="center", fontsize=14)
ax.tick_params(axis="both", labelbottom=False, bottom = False, labelleft=False, left=False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)