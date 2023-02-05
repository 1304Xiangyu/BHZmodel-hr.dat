import numpy as np
from numpy import linalg
import math 
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 


print("BHZ model")

u = -1.2

a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([0.0, 1.0, 0.0])
a3 = np.array([0.0, 0.0, 1.0])

volume = a1[0]*(a2[1]*a3[2] - a2[1]*a3[0])

#R1,R2,R3 = 2,2,1
NRPTS = 9

irvec = np.zeros((3,NRPTS),dtype=int)
 
l,m,n,N = -1,-1,0,0
while l<=1:
    while m<=1:
        while n<=0:
            irvec[0,N] = l
            irvec[1,N] = m
            irvec[2,N] = n 
            N = N+1
            n = n+1
        m = m+1
        n = 0
    l = l+1
    m = -1

Nk1 = 16
Nk2 = 16

# It is a square lattice model

l,m = 0, 0

kx = np.zeros(Nk1)
ky = np.zeros(Nk2)
while l<= Nk1-1:
  kx[l] = -math.pi*a2[1]*a3[2]/volume+2*l*math.pi*a2[1]*a3[2]/(volume*Nk1)  # in the direction of \vector{i}
  #print(kx[l])
  l = l+1

while m<=Nk2-1:
  ky[m] = -math.pi*a2[1]*a1[0]/volume+2*m*math.pi*a3[2]*a1[0]/(volume*Nk2)  # in the direction of \vector{j}
  m = m+1
  

Hk = np.zeros((4,4,Nk1,Nk2),dtype=complex)
#Hk_trans = np.zeros((4,4,Nk1,Nk2),dtype=complex)
#print(Hk[:,:,0,0])

#Hk[:,:,0,0] = ([2,2],[1,1])
#print(Hk[:,:,0,0])

l,m = 0,0
while l<=Nk1-1:
    while m<=Nk2-1:
        Hk[:,:,l,m] = ([u+math.cos(kx[l])+math.cos(ky[m]), -math.sin(ky[m])*1j+ math.sin(kx[l]), 0, 0.3],
              [math.sin(ky[m])*1j+math.sin(kx[l]), -(u+math.cos(kx[l])+math.cos(ky[m])), 0.3, 0],
              [0, 0.3, u+math.cos(kx[l])+math.cos(ky[m]), -math.sin(ky[m])*1j-math.sin(kx[l])],
              [0.3, 0, math.sin(ky[m])*1j-math.sin(kx[l]), -(u+math.cos(kx[l])+math.cos(ky[m]))])
        
        #Hk_trans = 
        m = m+1
    l = l+1
    m = 0


    
l,m = 0,0
eigenvalue = np.zeros((4,Nk1,Nk2))
#eigenvalue_ascend = np.zeros((4,Nk1,Nk2))
eigenvec   = np.zeros((4,4,Nk1,Nk2),dtype=complex)
while l<=Nk1-1:
    while m<=Nk2-1: 
        eigenvalue[:,l,m],eigenvec[:,:,l,m] = np.linalg.eig(Hk[:,:,l,m]) 
        eigenvalue[:,l,m]=np.sort(eigenvalue[:,l,m])
        #print(eigenvalue[l,:])
        #print(eigenvec[l,:,:])
        m = m+1
    l = l+1
    m = 0

l = 0
#while l<=Nk1-1:
#    print(eigenvalue[:,l,2])
#    l = l+1
    

# Do the Fourier Transformation
# Hmn(R) = \sum_k exp(-ikR)Hmn(k)/(num_kpts)
l,m,n= 0,0,0
kdotR = 0
HmnR = np.zeros((4,4,NRPTS),dtype=complex)
factor = 0
while l<=NRPTS-1:
    R = irvec[0,l]*a1 + irvec[1,l]*a2 + irvec[2,l]*a3
    #print('R')
    #print(R)
    while m<=Nk1-1:
        while n<=Nk2-1:
            kdotR = kx[m]*R[0]+ky[n]*R[1]
            factor = cmath.exp(-1j*kdotR)
            #print(kdotR)
            
            HmnR[:,:,l] = factor*Hk[:,:,m,n]/(Nk1*Nk2) + HmnR[:,:,l]
            n = n+1
        m = m+1
        n = 0
    l = l+1
    m = 0
    
l = 0
#write HmnR into file
f = open('BHZ-symmetric_hr.dat','w')
f.write("Hr_dat for BHZ model with C = 0.3sigma_x"+'\n')      # Comment line
f.write("4"+'\n')                         # Number of Wannier Orbits
f.write(str(NRPTS) + '\n')                       # Number of R vectors
# Number of degeneracy
while l<=NRPTS-1:
    f.write(str(1) + '  ')
    l = l+1
f.write('\n')
    
# Hmn(R)
l, m, n = 0, 0, 0
while l<=NRPTS-1:
    while m<=3:
        while n<=3:
            f.write("{:8d}{:8d}{:8d}".format(irvec[0,l],irvec[1,l],irvec[2,l]))
            f.write("{:8d}{:8d}".format(m+1,n+1))
            f.write("{:20.10f}{:20.10f}\n".format(np.real(HmnR[m,n,l]), np.imag(HmnR[m,n,l])))
            n = n+1
        m = m+1
        n = 0
    l = l+1
    m = 0
    
f.close()

fig1 = plt.figure(1)
plt.plot(kx[:],eigenvalue[0,:,2],'k')
plt.plot(kx[:],eigenvalue[1,:,2],'k')
plt.plot(kx[:],eigenvalue[2,:,2],'k')
plt.plot(kx[:],eigenvalue[3,:,2],'k')
plt.savefig("kx_band-symmetric.png")
plt.close()
#plt.show()

fig2 = plt.figure(2)
kx,ky = np.meshgrid(kx,ky)
ax = plt.axes(projection='3d')
ax.plot_surface(kx,ky,eigenvalue[1,:,:])
ax.plot_surface(kx,ky,eigenvalue[2,:,:])
#ax4.plot_surface(kx,ky,eigenvalue[:,:,0])
plt.savefig("band_surf-symmetric.png")
    


              




