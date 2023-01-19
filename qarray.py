from .imports import *



class qarray(np.ndarray):
    """
    Class that emulates numpy ndarray but using complex numbers by default.
    
    Attributes
    ----------
    None.
    
    Methods
    ----------
    dagg():
        Dagger
    eig():
        Eigen
    tri_to_herm():
        Transdorms into hermitian.
    
    """
    def __new__(cls, a):
        """
        Creates an array of complex numbers.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        obj : TYPE
            DESCRIPTION.

        """
        obj = np.array(a, dtype=np.complex128).view(cls)
        return obj
    
    def dagg(self):
        """
        Computes the hermitian conjugate of shaped (...,N,N) arrays.

        Returns
        -------
        ndarray
            Hermitian conjugate.

        """
        return self.T.conj()
    
    def eig(self, tol=0):
        """
        Computes the eigenvalues and eigenvectors of shaped (...,N,N) arrays.

        Parameters
        ----------
        tol : float, optional
            Tolerance under numerical errors. The default is 0.

        Returns
        -------
        vals : ndarray
            Eigenvalues.
        vects : ndarray
            Eigenvectors.

        """
        vals,vects = np.linalg.eig(self)
        vals *= abs(vals)>=tol
        vects *= abs(vects)>=tol
        return vals,vects
    
    def tri_to_herm(self, mode="U"):
        """
        Transform the own matrix from triangular to hermitian.

        Parameters
        ----------
        mode : str, optional
            Defines the filled triangle, "U" for upper and "L" for lower. The default is "U".

        Returns
        -------
        None.

        """
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
    """
    Constructs a qarray full of zeros.

    Parameters
    ----------
    sh : tuple, list
        Shape.

    Returns
    -------
    quarray
        Qarray of zeros.

    """
    N = np.prod(sh)
    return qarray([0]*N).reshape(sh)


def qidentity(sz):
    """
    Creates an identity matrix.

    Parameters
    ----------
    sz : int
        Matrix size.

    Returns
    -------
    I : qarray
        Identity.

    """
    I = qzeros((sz,sz))
    for i in range(sz):
        I[i,i] = 1
    return I


def qbasis(k):
    """
    Generates k canonical vectors.

    Parameters
    ----------
    k : int
        Dimension.

    Returns
    -------
    tuple
        Canonical qvectors.

    """
    return tuple(map(tuple, qidentity(k)[None,...,None]))[0]


def qmat_el(v1, v2, M):
    """
    Compute the projection over v2 of v1 after apply a matrix M.

    Parameters
    ----------
    v1 : qarray
        Vector.
    v2 : qarrya
        Vector.
    M : qarray
        Vector.

    Returns
    -------
    float
        Projection.

    """
    return (v1.dagg() @ M @ v2)[0,0]


def func_op(func, A):
    """
    Computes the function of an operator.

    Parameters
    ----------
    func : function
        One argument function applied over the operator.
    A : qarray
        Operator.

    Returns
    -------
    qarray
        f(A)

    """
    N = A.shape[0]
    a,U = np.linalg.eig(A)
    return U @ (func(a)*qidentity(N)) @ U.dagg()