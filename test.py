from .imports import *
from .params import *


class Test:
    def __init__(self):
        self.mat = np.zeros((2,2))
    
    @property
    def u(self):
        return gp
    @u.setter
    def u(self, x):
        gp = x
        self.mat[0,0] = x
        self.mat[1,1] = x
    
    @property
    def v(self):
        return gs
    @v.setter
    def v(self, x):
        gs = x
        self.mat[0,1] = x
        self.mat[1,0] = x
