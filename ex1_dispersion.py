#!/usr/bin/env python

'''
Example 1: multi-mode dispersion with and without a water layer
Ethan Williams
2023-10-31

To increase the accuracy at low frequencies, increase the model depth (zfac)
The find more modes at each frequency, increase the search density (nb)
The number of points in the integration (nz) is less important 
'''

from direct_integration import *
from cps_wrapper import query_cps
import numpy as np
import matplotlib.pyplot as plt

# layered half-space
d = [50,100,350,500]
vp = [2000,2500,3500,5500]
vs = [1000,1400,2200,3500]
ro = [2600,2800,3000,3200]
# add water layer
h = 500
vw = 1500
rw = 1000
nh = 10
# create velocity_model
M1 = velocity_model(d,vp,vs,ro)
M2 = velocity_model(d,vp,vs,ro,True,h,vw,rw,nh)

# range of frequencies
nf = 20
fs = np.logspace(0,1,nf)

# calculate all modes
cp_nowater = []
cp_water = []
for i,f in enumerate(fs):
    print('Working on %.2f Hz (%d/%d)' % (f,i+1,nf))
    cp_nowater.append(dispersion(f,M1,nb=30,nz=200,zfac=3,all_modes=True,adaptive_depth=True))
    cp_water.append(dispersion(f,M2,nb=30,nz=200,zfac=3,all_modes=True,adaptive_depth=True))

# CPS no water layer
vp_ = 1e-3*M1.vp
vs_ = 1e-3*M1.vs
ro_ = 1e-3*M1.rho
z_ = 1e-3*M1.z
T1, C1, U, UR, TR, UZ, TZ, I0, I1, I2, I3 = query_cps(vp_,vs_,ro_,z_,fs)

# CPS with water layer
vp_ = 1e-3*np.concatenate(([M2.vw],M2.vp),axis=0)
vs_ = 1e-3*np.concatenate(([0],M2.vs),axis=0)
ro_ = 1e-3*np.concatenate(([M2.rw],M2.rho),axis=0)
z_ = 1e-3*np.concatenate(([0],M2.z+M2.h),axis=0)
T2, C2, U, UR, TR, UZ, TZ, I0, I1, I2, I3 = query_cps(vp_,vs_,ro_,z_,fs)

# Plot comparison
fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,6))
for f,cp in zip(fs,cp_nowater):
    ax[0].semilogx(np.ones(len(cp))*f,cp,'ko')
ax[0].semilogx(1./T1,C1*1e3,'r^',label='CPS')
for f,cp in zip(fs,cp_water):
    ax[1].semilogx(np.ones(len(cp))*f,cp,'ko')
ax[1].semilogx(1./T2,C2*1e3,'r^',label='CPS')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Phase velocity (m/s)')
ax[0].set_title('No water layer')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_title('Water layer')

plt.show()

