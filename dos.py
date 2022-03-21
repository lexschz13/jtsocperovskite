from .imports import *
from .qarray import *

kB = 8.617333262e-5	


def dos(x, H, idxs, dump, T, spol):
    if type(x) is not np.ndarray:
        x = np.array([x])
    if type(spol) is not np.ndarray:
        spol = np.array(spol, dtype=np.float64)
    spol /= np.sum(spol)
    w,U = np.linalg.eigh(H)
    g_idxs = np.where(w < kB*T)[0]
    e_idxs = np.where(w >= kB*T)[0]
    
    gi,ig = np.meshgrid(g_idxs,idxs)
    ei,ie = np.meshgrid(e_idxs,idxs)
    G = np.sum(spol[:,None,None,None] * abs(U[ig,gi,None,None])**2 * abs(U[ie,ei][:,None,:,None])**2/(x[None,None,None,:]-w[None,None,e_idxs,None].real + w[None,g_idxs,None,None] + 1j*dump), axis=(0,1,2))
    return -G.imag/np.pi
