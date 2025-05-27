from .imports import *
from scipy.spatial.transform import Rotation as R


def icdf(x, k=1):
    """
    Computes inverse cumulative distribution function of Kent's distribution with no ellipticity and centered at north pole.

    Parameters
    ----------
    x : float, ndarray
        Numbers on the interval [0,1].
    k : float, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    float, ndarray
        Inverse cumulative distribution.

    """
    return np.arccos(k**(-1) * np.log(np.exp(k) - 2*np.sinh(k)*x))


def rd_vec(samples, k=1, axis=(0,0,1)):
    """
    Computes some random vectors sampled according Kent's distribution with no ellipticity.

    Parameters
    ----------
    samples : int
        Number of samples.
    k : float, optional
        Kent parameter. The default is 1.
    axis : array-like, optional
        Direction of the center of the distribution. The default is (0,0,1).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    import sys
    if (k > np.log(sys.float_info.max)) or (k == np.inf):
        # Large values of k colapse the distribution in one point.
        return np.array([axis for _ in range(samples)])
    elif k < 0:
        Error("Non valid value for Kent parameter")
        quit()
    elif k == 0:
        # k=0 is equivalent to uniform distribution.
        theta = np.random.random(samples)*np.pi
    else:
        # Polar angle is determined by inverse cumulative distribution function.
        theta = icdf(np.random.random(samples), k)
    phi = np.random.random(samples)*2*np.pi # Azimutal angle distribution is uniform.
    rad = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    axis_rot = np.array([-axis[1],axis[0],0])/np.sqrt(axis[0]**2+axis[1]**2 + (axis[0]+axis[1]==0))
    rot = R.from_rotvec(np.arccos(axis[2]/rad)*np.array(axis_rot)) # Rotation to center samples in 
    return rot.apply(np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T)
