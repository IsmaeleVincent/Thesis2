import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

pi=np.pi
rad=pi/180

n_diff= 4 #number of peaks for each side, for example: n_diff=2 for 5 diffracted waves

lam= 3.5e-3 #incoming wavelenght in micrometers
LAM= 0.5 #grating constant in micrometers

b=2*pi/lam #beta value 
G=2*pi/LAM #grating vector amplitude

bcr1=8.0#scattering lenght x density
bcr2=2
bcr3=1
bcr4=0

phi=0 #phase shift
phi1=0
phi2=0

n_1 = bcr1*2*pi/b**2 #modulation amplitude
n_2 = bcr2*2*pi/b**2
n_3 = bcr3*2*pi/b**2
n_4 = bcr4*2*pi/b**2

d=78 #thickness
 
def k_jz(theta, j, G): #z component of k_j
    k_jz=b*(1-(np.sin(theta)-j*G/b)**2)**0.5
    return k_jz
def dq_j (theta, j, G):#phase mismatch
    return b*np.cos(theta) - k_jz(theta, j, G)

phirange=np.linspace(0,4*pi,100)
num=0
for phi1 in phirange:
    num+=1
    th=np.linspace(-0.015,0.015, 150) #incident angle theta
    S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
    sum_diff = np.zeros(len(th)) 
    for t in range(len(th)):
        A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
        for i in range(len(A[0])):
            A[i][i]=dq_j(th[t],i-n_diff,G)#main diagonal (sqrt(-1) inserted later) #b**2*(n_0**2-1)/(2*k_jz(th[t],i-n_diff,G))
            if(i+1<len(A[0])):
                A[i][i+1]=b**2*n_1/(2*k_jz(th[t],i-n_diff,G)) #+1 diagonal
                A[i+1][i]=b**2*n_1/(2*k_jz(th[t],i-n_diff,G)) #-1 diagonal
            if(i+2<len(A[0]) and bcr2!=0):
                A[i][i+2]=b**2*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G))  #+2 diagonal
                A[i+2][i]=b**2*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G))    #-2 diagonal
            if(i+3<len(A[0]) and bcr3!=0):
                A[i][i+3]=b**2*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G)) #+3 diagonal
                A[i+3][i]=b**2*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G)) #-3 diagonal
            if(i+4<len(A[0]) and bcr4!=0):
                A[i][i+4]=b**2*n_4*np.exp(-1j*phi2)/(2*k_jz(th[t],i-n_diff,G)) #+4 diagonal
                A[i+4][i]=b**2*n_4*np.exp(1j*phi2)/(2*k_jz(th[t],i-n_diff,G)) #-4 diagonal
        A=1j*A #Diff. equation matrix
        
        w,v = np.linalg.eig(A)
        # for i in range(len(w)):
        #     print("AV=wV", np.allclose(np.dot(A,v[:,i]),w[i]*v[:,i]))
        # print(, )
        # print(w,v)
        v0=np.zeros(2*n_diff+1)
        v0[n_diff]=1
        #print(v0)
        c = np.linalg.solve(v,v0)
        for i in range(len(w)):
            v[:,i]=v[:,i]*c[i]*np.exp(w[i]*d)
        
        for i in range(len(S[:,0])):
            S[i,t] = sum(v[i,:]) # complex wave amplitudes
        #print(np.allclose(np.dot(v, c), v0))
    eta = S.copy().real
    for t in range(len(th)):
        for i in range(2*n_diff+1):
            eta[i,t] = abs(S[i,t])**2*k_jz(th[t],i-n_diff,G)/(b*np.cos(th[t]))
        sum_diff[t]= sum(eta[:,t])
    
    fig = plt.figure(figsize=(8,3.5), dpi=200)#constrained_layout=True
    gs_t = GridSpec(6, 3, figure=fig,hspace=0,wspace=0)
    gs_r = GridSpec(2, 3, figure=fig,hspace=2,wspace=0, top=0.75, bottom=0.13, right=0.86)
    ax = [fig.add_subplot(gs_t[0:2,:-1]), 
          fig.add_subplot(gs_t[2:4,:-1]),
          fig.add_subplot(gs_t[4:6,:-1]),
          fig.add_subplot(gs_r[0,-1],projection='polar'),
          fig.add_subplot(gs_r[1,-1],projection='polar')]
    for i in range(len(ax)-3):
            ax[i].tick_params(axis="x", labelbottom=False, bottom = False)
            ax[i].yaxis.set_label_position("right")
    ax[-3].yaxis.set_label_position("right")
    ax[-1].tick_params(axis="y", labelbottom=False, bottom = False,labelleft=False, left = False)
    ax[-1].tick_params(axis="x", pad=-4)
    ax[-1].grid(False)
    # ax[-1].set_bbox(True)
    ax[-1].set_ylim([0,1])
    ax[-1].set_title("$\phi_2$", y=1.5, bbox=dict(facecolor='none', edgecolor='k'))
    ax[-1].set_xticks([0,pi/2,pi,pi*3/2])
    ax[-1].set_xticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$'])
    ax[-2].tick_params(axis="y", labelbottom=False, bottom = False,labelleft=False, left = False)
    ax[-2].tick_params(axis="x", pad=-4)
    ax[-2].set_xticks([0,pi/2,pi,pi*3/2])
    ax[-2].set_xticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$'])
    ax[-2].grid(False)
    ax[-2].set_ylim([0,1])
    ax[-2].set_title("$\phi_1$", y=1.5, bbox=dict(facecolor='none', edgecolor='k'))
    # ax[-1].spines["top"].set_visible(False)
    # ax[-1].spines["left"].set_visible(False)
    # ax[-1].spines["bottom"].set_visible(False)
    # ax[-1].spines["right"].set_visible(False)
    lims=np.array([[0,1], [0,0.85],[0,0.3]])
    ticks=np.array([[0.0,1], [0.0,0.7],[0.0,0.2]])
    ax[0].set_title("$\lambda=$"+str("%.1f" %(lam*1e3),)+" nm\t $b_c\Delta\\rho_1$="+str(str("%.1f" %(bcr1),))+" $\mu m^{-2}$\t $b_c\Delta\\rho_2$="+str(str("%.1f" %(bcr2),))+" $\mu m^{-2}$\t $b_c\Delta\\rho_3$="+str(str("%.1f" %(bcr3),))+" $\mu m^{-2}$", fontsize=8)
    ax[0].set_ylabel("Order 0")
    for i in range(0,3):
        ax[i].set_ylim(lims[i])
        ax[i].set_yticks(ticks[i])
        ax[i].set_ylabel("Order $\pm$"+str(i))
        ax[i].plot(th,eta[n_diff-i,:],"-k", label="Fit (-"+str(i)+")")
        if i>0: 
            # ax[i].plot(th,0*th+np.amax(eta[n_diff-i,:]),"--k")
            ax[i].plot(th,eta[n_diff+i,:],"-",color = (0.8,0,0), label="Fit (+"+str(i)+")")  
            # ax[i].plot(th,0*th+np.amax(eta[n_diff+i,:]),"--", color = (0.8,0,0))
        # ax[i].legend()
    ax[-3].set_xlabel("$\\theta$ (rad)")
    fig.text(0.06, 0.5, 'Diff. efficiency', va='center', rotation='vertical', fontsize=11)
    ax[-2].plot([0,phi],[0,1], "-k.")
    ax[-1].plot([0,phi1],[0,1], "-k.")
    plt.savefig('Phase/Phase'+str(num)+'.png', format='png',bbox_inches='tight')
    plt.close(fig)

def make_gif(frame_folder):
    frames = [Image.open('Phase/Phase'+str(j+1)+'.png') for j in range(len(phirange))]
    frame_one = frames[0]
    frame_one.save("my_awesome_2.gif", format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)
    
if __name__ == "__main__":
    make_gif("Phase")