from .imports import *
from .qarray import *


def dos(x, H, idxs, dump):
    w,U = np.linalg.eig(H)
    G = np.zeros(x.size,dtype=np.complex128)
    for i in idxs:
        #[eigen,frequency], sum over eigenstates
        G += np.sum(abs(U[i,:,None])**2/((x[None,:]-w[:,None]) + 1j*dump), axis=0)
    return -G.imag/np.pi
