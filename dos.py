from .imports import *
from .qarray import *


def dos(x, H, idxs, dump):
    if type(x) is not np.ndarray:
        x = np.array([x])
    w,U = np.linalg.eigh(H)
    w0 = w.min()
    G = qzeros(x.size)
    for i in idxs:
        G += np.sum(abs(U[:,i,None])**2/(x[None,:]-w[:,None].real + w0 + 1j*dump), axis=0)
    return -G.imag/np.pi