#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:18:39 2022

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:28:47 2022

@author: aaa
"""




from scipy.special import erfc
from scipy.stats import exponnorm
from scipy.stats import skewnorm
from scipy.optimize import curve_fit as fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import inspect, os, time
def rho1(x, B, A, x0, sk, sig):
    g = exponnorm(loc=x0, K=sk, scale=sig)
    return B+A*g.pdf(x)
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]

tau = 0.5
mu = 1.4e-3
sigma = 0.7e-3
x = exponnorm.ppf(np.arange(0.01, 0.99, 1e-2), K=tau, loc=mu, scale=sigma)
print("x[0] = ",x[0])
plt.plot(x, rho1(x, 0, 1, mu, tau, sigma), "k.-")
Dx = x[1:]-x[:-1]
print("Dx avg=",np.average(Dx),"; Dx min=", np.amin(Dx),"; Dx max=", np.amax(Dx))
# def plot_func(th, bcr1, bcr2, bcr3, mu1, sigma, tau):
#     d=d0/np.cos((tilt[k]+zeta0)*rad)
#     wl=exponnorm.ppf(np.arange(0.01,0.99,wlp),K=tau, loc=mu1, scale=sigma)
#     a=rho(wl,tau, mu1, sigma)/sum(rho(wl,tau, mu1, sigma))
#     S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
#     eta=S.copy().real
#     eta_aus=eta.copy()
#     sum_diff = np.zeros(len(th))
#     for l in range(len(wl)):
#         lam=wl[l] #single wavelenght in micrometers
#         b=2*pi/lam #beta value 
#         n_1 = bcr1*2*pi/b**2
#         n_2 = bcr2*2*pi/b**2
#         n_3 = bcr3*2*pi/b**2
#         for t in range(len(th)):
#             A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
#             for i in range(len(A[0])):
#                 A[i][i]=-dq_j(th[t],i-n_diff,G,b)
#                 if(i+1<len(A[0])):
#                     A[i][i+1]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
#                     A[i+1][i]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
#                 if(i+2<len(A[0]) and bcr2!=0):
#                     A[i][i+2]=-b**2*n_0*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
#                     A[i+2][i]=-b**2*n_0*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
#                 if(i+3<len(A[0]) and bcr3!=0):
#                     A[i][i+3]=b**2*n_0*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
#                     A[i+3][i]=b**2*n_0*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
#             A=-1j*A
#             w,v = np.linalg.eig(A)
#             v0=np.zeros(2*n_diff+1)
#             v0[n_diff]=1
#             c = np.linalg.solve(v,v0)
#             for i in range(len(w)):
#                 v[:,i]=v[:,i]*c[i]*np.exp(w[i]*d)
#             for i in range(len(S[:,0])):
#                 S[i,t] = sum(v[i,:])
#         for t in range(len(th)):
#             for i in range(2*n_diff+1):
#                 eta_aus[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
#             sum_diff[t] = sum(eta[:,t])
#         eta+=eta_aus*a[l]
#     eta_ang = np.zeros((2*n_diff+1,len(diff_eff[:,0])))
#     x_int=np.arange(th[0],th[-1], 1e-6)
#     for i in range(n_diff*2+1):
#         f_int = interp1d(th,eta[i,:], kind="cubic")
#         conv=np.convolve(f_int(x_int),gauss_conv,mode="same")
#         f_int = interp1d(x_int,conv, kind="cubic")
#         eta_ang[i,:]=f_int(x*rad)
#     return eta_ang