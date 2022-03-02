from .imports import *



class qarray(np.ndarray):
    def __new__(cls, a):
        obj = np.array(a, dtype=np.complex128).view(cls)
        return obj
    
    def dagg(self):
        return self.T.conj()
    
    def eig(self, tol=0):
        vals,vects = np.linalg.eig(self)
        vals *= abs(vals)>=tol
        vects *= abs(vects)>=tol
        return vals,vects
    
    def tri_to_herm(self, mode="U"):
        if self.shape[-1] != self.shape[-2]:
            pass
        else:
            N = self.shape[-1]
            #Inspection of upper part
            #i is row, j is column
            for i in range(N):
                for j in range(i+1,N):
                    if mode=="U": #Upper mode, upper part is assumed to be filled
                        self[...,j,i] = self[...,i,j].conj()
                    elif mode=="L": #Lower mode, lower part is assumed to be filled
                        self[...,i,j] = self[...,j,i].conj()


def qzeros(sh):
    N = np.prod(sh)
    return qarray([0]*N).reshape(sh)


def qidentity(sz):
    I = qzeros((sz,sz))
    for i in range(sz):
        I[i,i] = 1
    return I


def qbasis(k):
    return tuple(map(tuple, qidentity(k)[None,...,None]))[0]


def qmat_el(v1, v2, M):
    return (v1.dagg() @ M @ v2)[0,0]
