import numpy as np
import matplotlib.pyplot as plt
pi=np.pi
rad=pi/180

sorted_fold_path="/home/aaa/Desktop/Thesis/Script/Trial/Sorted data/" #insert folder of sorted meausements files
allmeasurements = sorted_fold_path+"All measurements/"
allrenamed = allmeasurements +"All renamed/"
allmatrixes = allmeasurements + "All matrixes/"
allpictures = allmeasurements + "All pictures/"
allrawpictures = allmeasurements + "All raw pictures/"
alldata_analysis = allmeasurements + "All Data Analysis/"
allcropped_pictures = alldata_analysis + "All Cropped Pictures/"
allcontrolplots = alldata_analysis + "All Control plots/"
allcontrolfits = alldata_analysis + "All Control Fits/"
tiltangles=[0,40,48,61,69,71,79,80,81]
foldername=[]
for i in range(len(tiltangles)):
    foldername.append(str(tiltangles[i])+"deg")
foldername.append("79-4U_77c88deg")
foldername.append("79-8U_76c76deg")
foldername.append("79-12U_75c64deg")
foldername.append("79-16U_74c52deg")
tilt=[0,40,48,61,69,71,79,80,81,79,79,79,79]
n_theta=[26,46,28,17,16,20,21,20,19,48,43,59,24]  #number of measurements files for each folder (no flat, no compromised data)
n_pixel = 16384 #number of pixels in one measurement
"""
This block fits the diffraction efficiencies n(x)= n_0 + n_1 cos(Gx)
"""

n_diff= 7 #number of peaks for each side, for example: n_diff=2 for 5 diffracted waves
lam= 4e-3 #incoming wavelenght in micrometers
LAM= 0.5 #grating constant in micrometers
b=2*pi/lam #beta value 
G=2*pi/LAM
bcr1=8.0#scattering lenght x density
bcr2=2.
bcr3=0.
bcr4=0
n_0 =1. #average refractive index in the grating
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


th=np.linspace(-0.015,0.015, 1000) #incident angle theta
S=np.zeros((2*n_diff+1,len(th)),dtype=complex)
sum_diff = np.zeros(len(th)) 
for t in range(len(th)):
    A = np.zeros((2*n_diff+1,2*n_diff+1), dtype=complex)
    for i in range(len(A[0])):
        A[i][i]=dq_j(th[t],i-n_diff,G)#main diagonal (sqrt(-1) inserted later) #b**2*(n_0**2-1)/(2*k_jz(th[t],i-n_diff,G))
        if(i+1<len(A[0])):
            A[i][i+1]=-b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G)) #+1 diagonal
            A[i+1][i]=-b**2*n_0*n_1/(2*k_jz(th[t],i-n_diff,G)) #-1 diagonal
        if(i+2<len(A[0]) and bcr2!=0):
            A[i][i+2]=-b**2*n_0*n_2*np.exp(-1j*phi)/(2*k_jz(th[t],i-n_diff,G))  #+2 diagonal
            A[i+2][i]=-b**2*n_0*n_2*np.exp(1j*phi)/(2*k_jz(th[t],i-n_diff,G))    #-2 diagonal
        if(i+3<len(A[0]) and bcr3!=0):
            A[i][i+3]=-b**2*n_0*n_3*np.exp(-1j*phi1)/(2*k_jz(th[t],i-n_diff,G))
            A[i+3][i]=-b**2*n_0*n_3*np.exp(1j*phi1)/(2*k_jz(th[t],i-n_diff,G))
        if(i+4<len(A[0]) and bcr4!=0):
            A[i][i+4]=-b**2*n_0*n_4*np.exp(-1j*phi2)/(2*k_jz(th[t],i-n_diff,G))
            A[i+4][i]=-b**2*n_0*n_4*np.exp(1j*phi2)/(2*k_jz(th[t],i-n_diff,G))
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

fig, ax = plt.subplots(4,figsize=(10,10))
ax[0].plot(th,eta[n_diff,:])  
for i in range(1,4):
    ax[i].plot(th,eta[n_diff-i,:])
    ax[i].plot(th,eta[n_diff+i,:])   
