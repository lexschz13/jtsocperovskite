# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:48:06 2022

@author: asanchez3
"""

from .imports import *
from scipy.spatial.transform import Rotation as R


def icdf(x, k=1):
    return np.arccos(k**(-1) * np.log(np.exp(k) - 2*np.sinh(k)*x))


def rd_vec(samples, k=1, axis=(0,0,1)):
    import sys
    if (k > np.log(sys.float_info.max)) or (k == np.inf):
        return np.array([axis for _ in range(samples)])
    elif k < 0:
        Error("Non valid value for Kent parameter")
        quit()
    elif k == 0:
        theta = np.random.random(samples)*np.pi
    else:
        theta = icdf(np.random.random(samples), k)
    phi = np.random.random(samples)*2*np.pi
    rad = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    axis_rot = np.array([-axis[1],axis[0],0])/np.sqrt(axis[0]**2+axis[1]**2 + (axis[0]+axis[1]==0))
    rot = R.from_rotvec(np.arccos(axis[2]/rad)*np.array(axis_rot))
    return rot.apply(np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T)