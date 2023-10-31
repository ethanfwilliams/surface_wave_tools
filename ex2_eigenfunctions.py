#!/usr/bin/env python

'''
Example 2: fundamental mode dispersion and eigenfunctions
Ethan Williams
2023-10-31
'''

from direct_integration import *
from cps_wrapper import query_cps
import numpy as np
import matplotlib.pyplot as plt

# layered half-space
d = [50,100,1350,2500]
vp = [2000,2500,3500,5500]
vs = [1000,1400,2200,3500]
ro = [2600,2800,3000,3200]
# add water layer
h = 500
vw = 1500
rw = 1000
nh = 50
# create velocity_model
M = velocity_model(d,vp,vs,ro,True,h,vw,rw,nh)

# range of frequencies
fs = np.array([0.5,1,5])

# calculate fundamental mode
cs = []
ys = []
zs = []
for i,f in enumerate(fs):
    cp, yy, zz = dispersion(f,M,nb=10,nz=200,zfac=5,all_modes=False,adaptive_depth=True,return_eig=True)
    cs.append(cp)
    ys.append(yy)
    zs.append(zz)

# CPS with water layer
M2 = deepcopy(M)
zmax=2000; nz=100
M2.upsample_model(zmax/nz,zmax)
vp_ = 1e-3*np.concatenate((np.ones(M2.nh)*M2.vw,M2.vp),axis=0)
vs_ = 1e-3*np.concatenate((np.zeros(M2.nh),M2.vs),axis=0)
ro_ = 1e-3*np.concatenate((np.ones(M2.nh)*M2.rw,M2.rho),axis=0)
z_ = 1e-3*np.concatenate((np.arange(M2.nh)*M2.h/M2.nh,M2.z+M2.h),axis=0)
T, C, U, UR, TR, UZ, TZ, I0, I1, I2, I3 = query_cps(vp_,vs_,ro_,z_,fs)
F = 1./T
Z = z_[:-1]*1e3 - M2.h

# Normalize
for i in range(len(fs)):
    val = UZ[i,nh]
    UR[i,:] /= -val
    UZ[i,:] /= val
    TR[i,:] /= -val*1e-6
    TZ[i,:] /= val*1e-6
# Match ordering
F = F[::-1]
UR = UR[::-1,:]
UZ = UZ[::-1,:]
TR = TR[::-1,:]
TZ = TZ[::-1,:]

# Plot velocity model and eigenfunctions
for i in range(len(fs)):
    fig,ax = plt.subplots(1,5,sharey=True,figsize=(10,6))
    ax[0].plot(M2.vp,M2.z,'r',label='Vp')
    ax[0].plot(M2.vs,M2.z,'b',label='Vs')
    ax[0].invert_yaxis()
    ax[1].plot(UR[i,:],Z,'ro',label='CPS')
    ax[1].plot(ys[i][:,0],zs[i],'k',label='Integration')
    ax[2].plot(UZ[i,:],Z,'ro')
    ax[2].plot(ys[i][:,1],zs[i],'k')
    ax[3].plot(TR[i,:],Z,'ro')
    ax[3].plot(ys[i][:,2],zs[i],'k')
    ax[4].plot(TZ[i,:],Z,'ro')
    ax[4].plot(ys[i][:,3],zs[i],'k')
    ax[0].set_ylim([cs[i]/fs[i],-M.h])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Velocity (m/s)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_title('Velocity model')
    ax[1].set_xlabel('Displacement (m)')
    ax[1].set_title('Ur')
    ax[2].set_xlabel('Displacement (m)')
    ax[2].set_title('Uz')
    ax[3].set_xlabel('Stress (Pa)')
    ax[3].set_title('Tr')
    ax[4].set_xlabel('Stress (Pa)')
    ax[4].set_title('Tz')
    fig.suptitle('Eigenfuntions at %.2f Hz' % fs[i])

plt.show()

