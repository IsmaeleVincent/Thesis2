# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:34:28 2022

@author: ismae
48 line 70 theta 9
"""

from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import cmath
from scipy.stats import chisquare as cs
import matplotlib.patches as mpatches

plt.rcParams['hatch.linewidth'] = 5
plt.rcParams['hatch.color'] = (0.2,0.7,0.2)#"#33c233"#"#7a7afb"
plt.rcParams['font.size'] = 15
def distr(x,A1,A2,A3,x01,x02,x03, sx1,sx2,sx3,):
    return A1/(sx1)*np.exp(-0.5*(x-x01)**2/sx1**2)+A2/(sx2)*np.exp(-0.5*(x-x02)**2/sx2**2)+A3/(sx3)*np.exp(-0.5*(x-x03)**2/sx3**2)

def gauss(x,A,x0,sx):
    return A/(sx)*np.exp(-0.5*(x-x0)**2/sx**2)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
f="Overlap.csv"
data = np.loadtxt(f, skiprows=1, delimiter=",")
data[:,0]*=2
ymax = np.amax(data[:,1])
ymin = np.amin(data[:,1])
P0=[1000,600,100,0,20,30,2,2,3]
print(data[:,0])
#data[:,0] = (data[:,0]-xmax+10)*2e-3
#data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
print("p=",p)
print("cov=", np.diag(cov)**0.5)
xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
ax.set_xlabel("mm")
ax.set_ylabel("Counts")
xg=np.zeros((3,2),dtype=int)

for j in range(3):
    if j>0:
        xg[j,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]-p[3+j])<=3*p[6+j]][0])[0][0]
        xg[j,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]-p[3+j])<=3*p[6+j]][-1])[0][0]+1
    else:
        xg[j,0]=np.where(data[:,0]==data[:,0][abs(data[:,0]-p[3+j])<=3.5*p[6+j]][0])[0][0]
        xg[j,1]=np.where(data[:,0]==data[:,0][abs(data[:,0]-p[3+j])<=3.5*p[6+j]][-1])[0][0]+1
frac=np.zeros((len(data[xg[2,0]:xg[1,1],0])+1,2))
for j in range(xg[2,0],xg[1,1]+1):
    frac[j-xg[2,0],0]=data[j,1]*gauss(data[j,0], p[1], p[4], p[7])/(gauss(data[j,0], p[1], p[4], p[7])+gauss(data[j,0], p[2], p[5], p[8]))#gauss(data[j,0], p[1], p[4], p[7])#
    frac[j-xg[2,0],1]=data[j,1]*gauss(data[j,0], p[2], p[5], p[8])/(gauss(data[j,0], p[1], p[4], p[7])+gauss(data[j,0], p[2], p[5], p[8]))#gauss(data[j,0], p[2], p[5], p[8])#
# ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "g-", alpha=0.5,label="1st order")
# ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "b-", alpha=0.5, label="2nd order")
# ax.plot(data[:,0]-xmax,data[:,1], "k^", label="Measurement")
# ax.plot(xplt-xmax, distr(xplt,*p), "k--", label="Profile Fit")
ax.plot(data[:,0]-xmax,data[:,1], "k^", label="Measurement")
ax.set_ylim([0,800])
# ax.plot(xplt1-xmax,(xplt1-xmax)*0 , 'r-|', label="0th order")
ax.fill_between(data[xg[0,0]:xg[0,1],0]-xmax, data[xg[0,0]:xg[0,1],1],color = (0.7,0.2,0.2),ec="none", label="0th order")
ax.fill_between(data[xg[0,1]:xg[2,0],0]-xmax, data[xg[0,1]:xg[2,0],1],color = (0.2,0.7,0.2), label="1st order\nno overlap")
ax.fill_between(data[xg[2,0]:xg[1,1],0]-xmax, data[xg[2,0]:xg[1,1],1],fc=(0.2,0.2,0.7) ,ec= (0.2,0.7,0.2) ,hatch="//", label="Overlap region")#7fbf7f
ax.fill_between(data[xg[1,1]:,0]-xmax, data[xg[1,1]:,1],color = (0.2,0.2,0.7), label="2nd order\nno overlap")
# ax.fill_between(data[xg[2,0]:xg[1,1],0]-xmax, frac[:-1,0],color= "g", alpha= 0.5,hatch="//", label="Overlap 1st order")
# ax.fill_between(data[xg[2,0]:xg[1,1],0]-xmax, frac[:-1,1],color= "b", alpha= 0.5,hatch="//", label="Overlap 2nd order")
# ax.plot(data[xg[1,0]:xg[2,1],0],frac[:,0],"go")
# ax.plot(data[xg[1,0]:xg[2,1],0],frac[:,1],"ko")
# ax.plot(xplt0-xmax, gauss(xplt0,*p[0:3]), "r^",  label="0th order")

# ax.plot(xplt1, xplt1*0 +distr(p[1],*p[0:2],0), "k-|")
# ax.vlines(0,0,distr(p[1],*p[0:2],0), ls="--",color="grey")
# ax.vlines(p[1],0,distr(p[1],*p[0:2],0), ls="--", color="grey")
#ax.text(-0.5, 225, "$\sigma \\approx $"+str("%.2f"%p[1])+" mm", fontsize="large",color="k",backgroundcolor="white")
plt.legend(loc=0)
# plt.savefig('Overlap.eps', format='pdf',bbox_inches='tight')

# f="Overlap.csv"
# data = np.loadtxt(f, skiprows=1, delimiter=",")
# data[:,0]*=2
# ymax = np.amax(data[:,1])
# ymin = np.amin(data[:,1])
# P0=[1000, 2,30, 600, 2, 20, 100, 3,10]
# print(data[:,0])
# #data[:,0] = (data[:,0]-xmax+10)*2e-3
# #data[:,1] = (data[:,1]-ymin)/(ymax-ymin)
# p,cov=fit(distr,data[:,0],data[:,1],p0=P0)
# print("p=",p)
# print("cov=", np.diag(cov)**0.5)
# xplt=np.linspace(data[:,0][0], data[:,0][-1], 1000)
# xmax = xplt[distr(xplt,*p)==np.amax(distr(xplt,*p))]
# ax.set_xlabel("mm")
# ax.set_ylabel("Counts")
# # ax.plot(xplt-xmax, gauss(xplt,*p[0:3]), "r-", label="0th order")
# # ax.plot(xplt-xmax, gauss(xplt,*p[3:6]), "g-", label="1st order")
# # ax.plot(xplt-xmax, gauss(xplt,*p[6:9]), "b-", label="2nd order")
# ax.plot(data[:,0]-xmax,data[:,1], "k-^", label="Measurement")
# ax.text(-0.5, 120, "0", fontsize="large",color="k",backgroundcolor="white")
# ax.text(12, 120, "+1", fontsize="large",color="k",backgroundcolor="white")
# ax.text(21, 120, "+2", fontsize="large",color="k",backgroundcolor="white")
# plt.legend(loc=0)
# ax.set_ylim([0,800])
plt.savefig('Overlap_dist.eps', format='pdf',bbox_inches='tight')