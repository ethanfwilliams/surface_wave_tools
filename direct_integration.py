# Rayleigh waves from direct integration
# Ethan Williams - 2023/10/27

import numpy as np
from copy import deepcopy
from scipy.integrate import odeint
from scipy.optimize import ridder

class layer:
    def __init__(self,d,vp,vs,rho):
        self.d = np.array(d)
        self.vp = np.array(vp)
        self.vs = np.array(vs)
        self.rho = np.array(rho)
        self.mu = self.rho * self.vs**2
        self.la = self.rho * self.vp**2 - 2*self.mu
        return
    
class velocity_model:
    '''
    container for velocity model
    water layer is handled with:
      is_water - True or False
      h - water depth (m)
      vw - water acoustic speed (m/s)
      rw - water density (kg/m^3)
      nh - number of points to evaluate in water
    '''
    def __init__(self,d,vp,vs,rho,is_water=False,h=None,vw=None,rw=None,nh=None):
        self.nl = len(d)
        self.layers = np.array([layer(d[i],vp[i],vs[i],rho[i]) for i in range(self.nl)])
        self.nz = self.nl+1
        self.vp = np.array(vp)
        self.vs = np.array(vs)
        self.rho = np.array(rho)
        self.z = np.concatenate(([0],np.cumsum(d)))
        self.mu = self.rho * self.vs**2
        self.la = self.rho * self.vp**2 - 2*self.mu
        self.is_water = is_water # True if water
        self.h = h # water layer thickness
        self.vw = vw # water layer sound speed
        self.rw = rw # water layer density
        self.nh = nh # number of points in water layer
        return

    def upsample_model(self,dz,zmax):
        '''
        upsample model or extrapolate to greater depth
        '''
        n = [int(np.ceil(l.d/dz)) for l in self.layers]
        zz = np.cumsum([l.d for l in self.layers])
        nz = np.sum(n)
        if nz<=self.nl:
            raise ValueError('Specified nz is less than number of layers...')
        vp_ = np.zeros(nz)
        vs_ = np.zeros(nz)
        ro_ = np.zeros(nz)
        z_ = np.zeros(nz)
        iz = 0
        for i,l in enumerate(self.layers):
            vp_[iz:iz+n[i]] = l.vp
            vs_[iz:iz+n[i]] = l.vs
            ro_[iz:iz+n[i]] = l.rho
            if i == 0:
                z_[iz:iz+n[i]] = np.linspace(0,l.d,n[i]+1)[:-1]
            else:
                z0 = zz[i-1]
                z_[iz:iz+n[i]] = np.linspace(z0,z0+l.d,n[i]+1)[:-1]
            iz += n[i]
        if max(z_) >= zmax:
            idz = (z_<=zmax)
            self.vp = vp_[idz]
            self.vs = vs_[idz]
            self.rho = ro_[idz]
            self.z = z_[idz]
        else:
            nfill = int(np.ceil((zmax-z_[-1])/dz))
            self.vp = np.concatenate((vp_,np.ones(nfill)*vp_[-1]),axis=0)
            self.vs = np.concatenate((vs_,np.ones(nfill)*vs_[-1]),axis=0)
            self.rho = np.concatenate((ro_,np.ones(nfill)*ro_[-1]),axis=0)
            self.z = np.concatenate((z_,np.linspace(z_[-1]+dz,zmax,nfill)),axis=0)
        self.nz = len(self.z)
        self.d = np.diff(self.z)
        self.mu = self.rho * self.vs**2
        self.la = self.rho * self.vp**2 - 2*self.mu
        return

def displacement_stress(y,zi,k,w,m):
    '''
    y - displacement-stress vector for Rayleigh waves
        y[0] = ux; y[1] = uz; y[2] = tx; y[3] = tz
    zi - layer depth to evaluate velocity model
    k - angular wavenumber (rad/m)
    w - angular frequency (rad/s)
    m - velocity_model object
    returns dy/dz = A@y
    '''
    iz = np.argmin(abs(m.z-zi))
    ro = m.rho[iz]
    la = m.la[iz]
    mu = m.mu[iz]
    s = 4*mu*(la+mu)/(la+2*mu)
    A = np.zeros((4,4))
    A[0,:] = [0, k, 1./mu, 0]
    A[1,:] = [-k*la/(la+2*mu), 0, 0, 1./(la+2*mu)]
    A[2,:] = [s*k**2 - ro*w**2, 0, 0, k*la/(la+2*mu)]
    A[3,:] = [0, -ro*w**2, -k, 0]
    return A @ y

def integrate(k,w,m):
    '''
    k - angular wavenumber (rad/m)
    w - angular frequency (rad/s)
    m - velocity_model object
    returns two eigenfunction solutions
    '''
    if k**2 <= w**2/m.vs[-1]**2:
        print('Wavenumber too large for bottom half-space')
    # Vertical wavenumbers
    va = np.sqrt(k**2 - w**2/m.vp[-1]**2)
    vb = np.sqrt(k**2 - w**2/m.vs[-1]**2)
    # Starting solution in bottom half-space
    y0a = np.array([k, va, 2*k*m.mu[-1]*va, m.mu[-1]*(k**2+vb**2)]) * np.exp(va*m.z[-1])
    y0b = np.array([vb, k, m.mu[-1]*(k**2+vb**2), 2*k*vb*m.mu[-1]]) * np.exp(vb*m.z[-1])
    # Integrate to surface
    ya = odeint(displacement_stress,y0a,m.z[::-1],args=((k,w,m)))
    yb = odeint(displacement_stress,y0b,m.z[::-1],args=((k,w,m)))
    return ya[::-1,:], yb[::-1,:]

def eigenfunction(k,w,m,just_det=False):
    '''
    k - angular wavenumber (rad/m)
    w - angular frequency (rad/s)
    m - velocity_model object
    just_det - return determinant only, or determinant + eigenfunctions
    '''
    # Integrate from bottom half space to surface of solid Earth
    ya,yb = integrate(k,w,m)
    # If no water layer, return determinant and eigenfunction
    if not m.is_water:
        det = ya[0,2]*yb[0,3] - yb[0,2]*ya[0,3]
        y = ya + (-ya[0,2]/yb[0,2]) * yb
        y /= y[0,1] # normalize
        #det = y[0,3]
        z = m.z
    else: # If water layer, append propagator matrix first
        ew = np.emath.sqrt(w**2/m.vw**2 - k**2)
        dh = np.linspace(0,m.h,m.nh)[::-1]
        yaw = np.zeros((m.nh,4),dtype=np.complex_)
        ybw = np.zeros((m.nh,4),dtype=np.complex_)
        yaw[:,1] = np.cos(ew*dh)*ya[0,1] - (ew/m.rw/w**2)*np.sin(ew*dh)*ya[0,3]
        yaw[:,3] = (w**2*m.rw/ew)*np.sin(ew*dh)*ya[0,1] + np.cos(ew*dh)*ya[0,3]
        yaw[:,0] = (k/m.rw/w**2)*yaw[:,3]
        ybw[:,1] = np.cos(ew*dh)*yb[0,1] - (ew/m.rw/w**2)*np.sin(ew*dh)*yb[0,3]
        ybw[:,3] = (w**2*m.rw/ew)*np.sin(ew*dh)*yb[0,1] + np.cos(ew*dh)*yb[0,3]
        ybw[:,0] = (k/m.rw/w**2)*ybw[:,3]
        det = np.real(ya[0,2]*ybw[0,3] - yb[0,2]*yaw[0,3])
        ys = ya + (-ya[0,2]/yb[0,2]) * yb
        yw = yaw + (-ya[0,2]/yb[0,2]) * ybw
        y = np.concatenate((yw,ys),axis=0)
        y /= ys[0,1] # normalize
        y = np.real(y)
        z = np.concatenate((-1*dh,m.z))
    if just_det:
        return det
    else:
        return det, y, z

def dispersion(f,m,nb=10,nz=1000,zfac=5,all_modes=False,adaptive_depth=False,return_eig=False,return_model=False):
    '''
    get phase velocity at frequency
    f - single frequency (Hz)
    m - velocity_model object
    nb - number of intervals for root search
        smaller (e.g. 10) = faster, but higher modes may be patchy
        larger (e.g. 100) = slower, but reliable to high frequencies
    nz - number of depth points (if using adaptive_depth)
    zfac - number of wavelengths to truncate model in depth
    all_modes - just return first mode if False
    adaptive_depth - upsample or extend model (important for low-freq stability)
    return_eig - also return eigenfunctions
    return_model - also return adapted velocity model
    '''
    w = 2*np.pi*f
    M = deepcopy(m)
    # bounds for search
    kmin = 1.01*(w/M.vs[-1])
    kmax = w/(0.7*np.min(M.vs))
    # resample/cut/extend velocity model
    if adaptive_depth:
        zmax = zfac*(2*np.pi/kmax)
        M.upsample_model(zmax/nz,zmax)
        # recalculate kmin
        kmin = 1.01*(w/M.vs[-1])
    # get brackets for root search
    kb = np.linspace(kmin,kmax,nb)
    dets = np.zeros(nb)
    for ik,k in enumerate(kb):
        dets[ik] = eigenfunction(k,w,M,just_det=True)
    # get brackets with sign change
    diff = np.diff(np.sign(dets))
    brac = np.argwhere(abs(diff)>1).flatten()
    # find root(s)
    if all_modes:
        ks = []
        for ib in range(len(brac)):
            a = kb[brac[ib]]
            b = kb[brac[ib]+1]
            k = ridder(eigenfunction,a,b,(w,M,True))
            ks.append(k)
        ks = np.array(ks)[::-1]
        cp = w/ks
    elif len(brac)>0:
        a = kb[brac[-1]]
        b = kb[brac[-1]+1]
        k = ridder(eigenfunction,a,b,(w,M,True))
        cp = w/k
    else:
        cp = np.nan
    # return eigenfunctions
    if return_eig:
        dh = M.h/M.nh
        zz = np.concatenate((-1*np.linspace(dh,M.h,M.nh)[::-1],M.z),axis=0)
        if all_modes:
            ys = np.zeros((len(ks),len(zz),4))
            for ik,k in enumerate(ks):
                _,y,z = eigenfunction(k,w,M,just_det=False)
                for j in range(4):
                    ys[i,:,j] = np.interp(zz,z,y[:,j])
        else:
            ys = np.zeros((len(zz),4))
            _,y,z = eigenfunction(k,w,M,just_det=False)
            for j in range(4):
                ys[:,j] = np.interp(zz,z,y[:,j])
        if return_model:
            return cp, ys, zz, M
        else:
            return cp, ys, zz
    else:
        if return_model:
            return cp, M
        else:
            return cp

# TO DO:
# - add energy integral and group velocity functions

