import numpy as np
from scipy.optimize import curve_fit as fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
pi=np.pi
rad=pi/180
plt.rcParams['font.size'] = 13
font_size=13
sorted_fold_path="/home/aaa/Desktop/Thesis2/Sorted data/" #insert folder of sorted meausements files
file_name=sorted_fold_path + "Safety plots/5RCWA_eta_TE_GS__0.5__d_322.4181085662225__e1_15.597184423005746__e2_2.339577663450862__e3_0.0__phi2_0.00__phi3_0.00.dat"
diff_eff=np.loadtxt(file_name)
def k_jz(theta, j, G,b):
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G,b):
    return b*np.cos(theta) - k_jz(theta, j, G, b)
def sin2(x,k):
    return np.sin(k*x)**2

n_diff= 4 #number of peaks for each side, for example: n=2 for 5 diffracted waves
P=[8,0,0]
fig = plt.figure(figsize=(7,3))#constrained_layout=True
gs = GridSpec(1, 1, figure=fig,wspace=0, hspace=-0.2, top=0.75)
ax = [fig.add_subplot(gs[0,0])]
lam=3e-3
LAM= 0.1 #grating constant in micrometers
G=2*pi/LAM
bcr1=P[0]#scattering lenght x density
bcr2=P[1]
bcr3=0
n_0 =1.
phi=P[2]*pi
phi1=0
d0=78
tilt=[0,40,48,61,69,71,79,80,81,77.88,76.76,75.64,74.52]
tilt=np.linspace(0,500,100)#np.sort(tilt)
pendol = np.zeros((len(tilt),5))
pendol[:,0] = tilt
k=-1
for zeta in tilt:
    k+=1
    d=zeta#d0/np.cos((zeta*rad))
    # pendol[k,0] = d
    th=[-0.015]#[-0.00893939]#[x[0]*rad-3*div,*x*rad,x[-1]*rad+3*div]
    S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
    eta=S.copy().real
    eta_aus=eta.copy()
    sum_diff = np.zeros(len(th))
    b=2*pi/lam #beta value 
    n_1 = bcr1*2*pi/b**2
    n_2 = bcr2*2*pi/b**2
    n_3 = bcr3*2*pi/b**2
    for t in [0]:#range(len(th2)):
        A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
        for i in range(len(A[0])):
            A[i][i]=-dq_j(th[t],i-n_diff,G,b)
            if(i+1<len(A[0])):
                A[i][i+1]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
                A[i+1][i]=b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G,b))
            if(i+2<len(A[0]) and bcr2!=0):
                A[i][i+2]=b**2*n_0*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
                A[i+2][i]=b**2*n_0*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G,b))
            if(i+3<len(A[0]) and bcr3!=0):
                A[i][i+3]=b**2*n_0*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
                A[i+3][i]=b**2*n_0*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G,b))
        A=-1j*A
        w,v = np.linalg.eig(A)
        v0=np.zeros(2*n_diff+1)
        v0[n_diff]=1
        c = np.linalg.solve(v,v0)
        for i in range(len(w)):
            v[:,i]=v[:,i]*c[i]*np.exp(w[i]*d)
        for i in range(len(S[:,0])):
            S[i,t] = sum(v[i,:])
    for t in range(len(th)):
        for i in range(2*n_diff+1):
            eta[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G,b)/(b*np.cos(th[t]))
        sum_diff[t] = sum(eta[:,t])
    pendol[k,1]=eta[n_diff,:]
    pendol[k,2]=eta[n_diff-1,:]
    # pendol[k,3]=eta[n_diff+1,:]
    # pendol[k,4]=eta[n_diff-3,:]
# ax[0].plot(eta[n_diff-1,:])
# tx=tilt[pendol[:,1]==np.min(pendol[:,1])]
# print(tx)
P[0]=n_1
P[1]=n_2
ax[0].plot(pendol[:,0],pendol[:,1],"-k", label="Order 0")   
ax[0].plot(pendol[:,0],pendol[:,2],"--k", label="Order 1")
#p,cov=fit(sin2, tilt, pendol[:,2],[1e-2])
#ax[0].plot(tilt[::3], sin2(tilt[::3],*p),"ok", label="$\sin ^2$")
# ax[0].plot(pendol[:,0],pendol[:,3],"--", color=(0.8,0,0),label="Order 2")
# ax[0].plot(pendol[:,0],pendol[:,4],"--", color=(0,0,0.5), label="Order 3")
ax[0].legend(ncol=1, framealpha=1, loc=5)
fig.suptitle("$\lambda$ = "+str(lam*1e3)+" $nm$   $\Lambda$ = "+str(LAM)+" $\mu m$   $\Delta n$ = "+str("%.1f" % (1e6*P[0],)) + "$\cdot 10^{-6}$", bbox=dict(fc=(1,1,1)), fontsize=font_size)
ax[0].set_title("$\\theta$ = $\\theta_{B1}$ $\\approx$ "+str("%.1f"%(-th[0]*1e2,))+"$\cdot 10^{-2}$ rad", fontsize=font_size)# ax[1].set_title("Parameters", fontsize=15)
ax[0].set_xlabel("Thickness ($\mu m$)")
ax[0].set_ylabel("Diff. efficiency")

plt.savefig('Pendel_simulation_bragg.eps', format='pdf',bbox_inches='tight')
plt.show()