from .imports import *
from .qarray import *
from .spin import S2x,S2y,S2z

kB = 8.617333262e-5





def dos(x, H, idxs, dump, T, ensemble):
    """
    Computes the spectral function for a given Hamiltonian and an ensemble.

    Parameters
    ----------
    x : array-like
        Frequencies.
    H : ndarray
        Hamiltonian.
    idxs : array-like
        Orbitals referred for ensemble.
    dump : float
        Dumping term.
    T : float
        Temperature.
    ensemble : array-like
        Ensemble.

    Returns
    -------
    ndarray
        Spectral function.

    """
    if type(x) is not np.ndarray:
        x = np.array([x])
    
    w,U = np.linalg.eigh(H)
    g_idxs = np.where((w-w[0]) <= kB*T)[0]
    e_idxs = np.where((w-w[0]) > kB*T)[0]
    
    non_zero_idxs = np.where(ensemble != 0)[0]
    non_zero_basis = np.array(idxs)[non_zero_idxs]
    
    if type(ensemble) is not np.ndarray:
        ensemble = np.array(ensemble, dtype=np.float64)
    
    ensemble /= non_zero_idxs.size
    
    G = np.sum(abs(ensemble[non_zero_idxs,None,None,None])**2 * abs(U[non_zero_basis,:][:,g_idxs][:,:,None,None])**2 * abs(U[non_zero_basis,:][:,e_idxs][:,None,:,None])**2/(x[None,None,None,:]-w[None,None,e_idxs,None] + w[None,g_idxs,None,None] + 1j*dump), axis=(0,1,2))
    return -G.imag/np.pi/g_idxs.size



def correl(x, H0, A, W, dump, T):
    """
    Unused
    """
    w,U = np.linalg.eigh(H0)
    w -= w[0]
    wn,wm = np.meshgrid(w,w)
    newA = U.dagg() @ A @ U
    newW = U.dagg() @ W @ U
    beta = 1/(kB*T)
    Z0 = np.sum(np.exp(-beta*w))
    retarded_filter = np.zeros((x.size,*A.shape))
    wh_neg_freq = np.where(wm>wn)
    retarded_filter[:,wh_neg_freq[0],wh_neg_freq[1]] += 1
    correl_terms = (1/Z0) * (np.exp(-beta*wn)-np.exp(-beta*wm))[None,:,:] * (newA * newW.T)[None,:,:] / (x[:,None,None] - (wm-wn)[None,:,:] + 1j*dump) * retarded_filter
    return np.sum(correl_terms, axis=(1,2))
