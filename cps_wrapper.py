# Simple wrapper for Rayleigh waves from CPS
# Ethan Williams - 2023/10/27

import os
import numpy as np

def query_cps(vp_,vs_,ro_,z_,f_):
    '''
    wrapper for CPS (Herrmann) sdisp96
    returns phase/group velocities, eigenfunctions, and energy integrals
    can accept array of frequencies (f_)
    only tested for one mode at a time (see -NMOD 1 below)
    may want to adjust root search (see -FACR 1 below)
    '''
    # Write frequency file
    with open('freq.txt','w') as fp:
        for freq in f_:
            fp.write('%.3f\n' % freq)
    # Get interval thicknesses
    d = np.diff(z_)
    nd = len(d)
    nf = len(f_)
    # Write velocity model file
    template = '\t %.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'
    with open('SCM.mod','w') as fp:
        fp.write('MODEL.01\n')
        fp.write('Example\n')
        fp.write('ISOTROPIC\n')
        fp.write('KGS\n')
        fp.write('FLAT EARTH\n')
        fp.write('1-D\n')
        fp.write('CONSTANT VELOCITY\n')
        fp.write('LINE08\n')
        fp.write('LINE09\n')
        fp.write('LINE10\n')
        fp.write('LINE11\n')
        fp.write('      H(KM)   VP(KM/S)   VS(KM/S) RHO(GM/CC)     QP'\
                + 'QS       ETAP       ETAS      FREFP      FREFS\n')
        for ii in range(nd): # cuts off bottom layer
            fp.write(template % (d[ii],vp_[ii],vs_[ii],ro_[ii],0,0,0,0,1,1))
    # Call CPS functions
    os.system('sprep96 -R -M SCM.mod -FARR freq.txt -NMOD 1 -FACR 1')
    os.system('sdisp96')
    os.system('sregn96 -DER -V > modes.txt')
    os.system('sdpder96 -R -C -TXT')
    # Read CPS outputs
    with open('SRDER.TXT','r') as fp:
        T = []
        C = []
        U = []
        UR = np.zeros((nf,nd))
        TR = np.zeros((nf,nd))
        UZ = np.zeros((nf,nd))
        TZ = np.zeros((nf,nd))
        starts = (3+nd+3) + np.arange(nf,dtype=int)*(nd+6)
        lno = 0
        iif = 0
        iid = 0
        for line in fp.readlines():
            if lno in starts:
                tmp = line.strip().split()
                T.append(float(tmp[2]))
                C.append(float(tmp[5]))
                U.append(float(tmp[8]))
            iif = int(np.floor((lno - (nd+9))/(nd+6)))
            iid = lno-(nd+9)-(nd+6)*iif
            if (iif >= 0) and (iid < nd):
                tmp = line.strip().split()
                UR[iif,iid] = float(tmp[1])
                TR[iif,iid] = float(tmp[2])
                UZ[iif,iid] = float(tmp[3])
                TZ[iif,iid] = float(tmp[4])
            lno += 1
    with open('modes.txt','r') as fp:
        I0 = []
        I1 = []
        I2 = []
        I3 = []
        starts = (nd+3) + np.arange(nf,dtype=int)*5
        lno = 0
        for line in fp.readlines():
            if lno in starts:
                tmp = line.strip().split()
                I0.append(float(tmp[1]))
                I1.append(float(tmp[3]))
                I2.append(float(tmp[5]))
                I3.append(float(tmp[7]))
            lno += 1
    T = np.array(T); C = np.array(C); U = np.array(U)
    I0 = np.array(I0); I1 = np.array(I1); I2 = np.array(I2)
    I3 = np.array(I3)
    os.system('rm SRDER.TXT modes.txt SCM.mod freq.txt sdisp96.dat  sdisp96.ray  SRDER.PLT  sregn96.der')
    return T, C, U, UR, TR, UZ, TZ, I0, I1, I2, I3


