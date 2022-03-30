from .imports import *
from .qarray import *
from .spin import S2x,S2y,S2z

kB = 8.617333262e-5





def dos(x, H, idxs, dump, T, ensemble, srot):
    if type(x) is not np.ndarray:
        x = np.array([x])
    
    Hrot = srot @ H @ srot.dagg()
    
    w,U = np.linalg.eigh(Hrot)
    g_idxs = np.where((w-w[0]) <= kB*T)[0]
    e_idxs = np.where((w-w[0]) > kB*T)[0]
    
    non_zero_idxs = np.where(ensemble != 0)[0]
    non_zero_basis = np.array(idxs)[non_zero_idxs]
    
    if type(ensemble) is not np.ndarray:
        ensemble = np.array(ensemble, dtype=np.float64)
    
    ensemble /= non_zero_idxs.size
    
    G = np.sum(abs(ensemble[non_zero_idxs,None,None,None])**2 * abs(U[non_zero_basis,:][:,g_idxs][:,:,None,None])**2 * abs(U[non_zero_basis,:][:,e_idxs][:,None,:,None])**2/(x[None,None,None,:]-w[None,None,e_idxs,None] + w[None,g_idxs,None,None] + 1j*dump), axis=(0,1,2))
    return -G.imag/np.pi/g_idxs.size
