from .imports import *
from .qarray import *


def dos(x, H, idxs, dump):
    w,U = H.eig()
    G = qzeros(x.size)
    for i in idxs:
        #[eigen,frequency], sum over eigenstates
        G += np.sum(abs(U[i,:,None])**2/((x[None,:]-w[:,None]) + 1j*dump), axis=0)
    return -G.imag/np.pi
