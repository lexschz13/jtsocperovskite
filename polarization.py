from .imports import *
from .qarray import *
from scipy.spatial.transform import Rotation as R


epsL = qarray([1,1j,0])*np.sqrt(0.5)
epsR = qarray([1,-1j,0])*np.sqrt(0.5)

def rot_eps(*cut):
    a,b,c = cut
    N = np.sqrt(a**2+b**2+c**2)
    axis = np.array([-b,a,0])/np.sqrt(a**2+b**2 + (a+b==0))
    r = R.from_rotvec(np.arccos(c/N)*axis)
    return r.apply(epsL), r.apply(epsR)
