from .imports import *
from .qarray import *
from scipy.spatial.transform import Rotation as R


epsL = qarray([1,1j,0])*np.sqrt(0.5) #Left circular
epsR = qarray([1,-1j,0])*np.sqrt(0.5) #Right circular

def rot_eps(*cut):
    """
    Computes LC and RC polarizations for a given propagation direction.

    Parameters
    ----------
    *cut : tuple/list pointer
        Propagation direction of light.

    Returns
    -------
    ndarray
        Left-circular polarization vector.
    ndarray
        Right-ciruclar polarization vector.

    """
    a,b,c = cut
    N = np.sqrt(a**2+b**2+c**2)
    axis = np.array([-b,a,0])/np.sqrt(a**2+b**2 + (a+b==0))
    r = R.from_rotvec(np.arccos(c/N)*axis)
    return r.apply(epsL), r.apply(epsR)
