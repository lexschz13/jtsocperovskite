from .imports import *
from .qarray import *
#from .angular_moms import Harmonics
#from .params import *
from .spin import *
from .hoppings import *
from .polarization import *
from .dos import *
from .random_spin import *


def defaults(dfl, name, kdict):
    """
    Set default values for key arguments.

    Parameters
    ----------
    dfl : any
        Default value.
    name : str
        Key.
    kdict : dict
        Dictionary with key arguments.

    Returns
    -------
    any
        If key is not in dictionary is default value, else is the correspondent dict value.

    """
    return dfl if not name in kdict else kdict[name]

N = 19
Ns = 8



class Solver:
    """
    Constructs a Hamiltonian and solve its spectral function for a given ensemmble.
    
    Attributes
    ----------
    delta : float
        Crystal field energy. Default is 2.
    FE : float
        Linear ExE Jahn-Teller parameter. Default is 0.75.
    GE : float
        Quadratic ExE Jahn-Teller parameter. Defualt is 0.02.
    FT : float
        Linear TxE Jahn-Teller parameter. Default is 0.15.
    dth : float
        Deviation from tetragonal elongation in Jahn-Teller effect as polar coordinate in mexican hat. Default is 0.
    xiSO : float
        Spin-orbit coupling amlplitude. Default is 0.02.
    kent : float
        Kent parameter for random distribution of the spins. Default is inf.
    tpd : float
        Hopping amplitude. Default is 0.03.
    CT : float
        Charge transfer energy. Default is 4.2.
    ssd : float
        Slater-Koster parameter sigma sd. Default is 0.87.
    sdd : float
        Slater-Koster parameter sigma dd. Default is 0.65.
    pdd : float
        Slater-Koster parameter pi dd. Default is 0.48.
    ddd : float
        Slater-Koster parameter delta dd. Default is 0.13.
    cut : array-like
        Propagation direction of light. Default is [0,0,1].
    dump : float
        Dumping parameter of spectral function. Default is 0.02.
    T : float
        Temperatuere. Default is 260. Default is [0,0,1].
    spin_dir : array-like
        Center if spin random directions. Default is [0,0,1].
    spin_pol : array-like
        Distribution of spin states with S=2 (M=-2,...,+2). Default is [1,1,1,1,1].
    
    Static attributes
    ----------
    Ns : int
        Number of atomic cells. Default is 8.
    Henergy : ndarray
        Crystal field Hamiltonian.
    HJT : ndarray
        Jahn-Teller Hamiltonian.
    HSOC : ndarray
        Spin-orbit coupling Hamiltonian.
    HhopL : ndarray
        Electromagnetic hopping effective Hamiltonian for left-circular polarization.
    HhopR : ndarray
        Electromagnetic hopping effective Hamiltonian for right-circular polarization.
    
    Methods
    ----------
    solve(x) :
        Computes the spectral function for a given frequencies.
    
    """
    def __init__(self, *args, **kwargs):
        """
        Constructs the Hamiltonian.

        Parameters
        ----------
        *args : tuple
            Unusefull.
        **kwargs : dict
            Dictionary with all non-static attributes of the class passed as key arguments.

        Returns
        -------
        None.

        """
        #Energies diag
        self.__delta     = defaults(2, "delta", kwargs)
        #Jahn-Teler
        self.__FE        = defaults(0.75, "FE", kwargs)
        self.__GE        = defaults(0.02, "GE", kwargs)
        self.__FT        = defaults(0.15, "FT", kwargs)
        self.__dth       = defaults(0, "dth", kwargs)
        #Spin-orbit
        self.__xiSO      = defaults(0.02, "xiSO", kwargs)
        self.__kent      = defaults(np.inf, "kent", kwargs)
        #Hopping
        self.__tpd       = defaults(0.03, "tpd", kwargs)
        self.__CT        = defaults(4.2, "CT", kwargs)
        self.__ssd       = defaults(0.87, "ssd", kwargs)
        self.__sdd       = defaults(0.65, "sdd", kwargs)
        self.__pdd       = defaults(0.48, "pdd", kwargs)
        self.__ddd       = defaults(0.13, "ddd", kwargs)
        #Polarization
        self.__cut       = defaults([0,0,1], "cut", kwargs)
        #Dumping
        self.__dump      = defaults(0.02, "dump", kwargs)
        #Propagation
        self.__T         = defaults(260, "temperature", kwargs)
        self.__spin_pol  = defaults([1,1,1,1,1], "spin_pol", kwargs)
        self.__spin_dir  = defaults([0,0,1], "spin_dir", kwargs)
        
        #Hamiltonian
        self.__Henergy = qzeros((Ns*N,Ns*N))
        self.__HJT = qzeros((Ns*N,Ns*N))
        self.__HSOC = qzeros((Ns*N,Ns*N))
        self.__HhopL = qzeros((Ns*N,Ns*N))
        self.__HhopR = qzeros((Ns*N,Ns*N))
        
        #Ensemble
        #self.__ensemble = self.__density_matrix(self.__spin_pol, self.__spin_dir)
        
        #Random spin
        if len(self.__spin_dir) == 8:
            self.__rd_spin = np.array(self.__spin_dir)
        else:
            self.__rd_spin = rd_vec(8, self.__kent, self.__spin_dir)
        # print(self.__rd_spin)
        # quit()
        
        #Construction
        for k in range(Ns):
            self.__Henergy[k*N:(k+1)*N,k*N:(k+1)*N] = self.__tan_sug(k)
            self.__HSOC[k*N:(k+1)*N,k*N:(k+1)*N] = self.__soc(self.__rd_spin[k,:])

        self.__HJT[0*N:1*N,0*N:1*N] = self.__jahn_teler(0, False)
        self.__HJT[1*N:2*N,1*N:2*N] = self.__jahn_teler(2, True)
        self.__HJT[2*N:3*N,2*N:3*N] = self.__jahn_teler(0, False)
        self.__HJT[3*N:4*N,3*N:4*N] = self.__jahn_teler(2, True)
        self.__HJT[4*N:5*N,4*N:5*N] = self.__jahn_teler(1, False)
        self.__HJT[5*N:6*N,5*N:6*N] = self.__jahn_teler(0, True)
        self.__HJT[6*N:7*N,6*N:7*N] = self.__jahn_teler(1, False)
        self.__HJT[7*N:8*N,7*N:8*N] = self.__jahn_teler(0, True)
        
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
        
    
    ################################################################################################
    #Definition of properties
    
    @property
    def delta(self):
        return self.__delta
    @delta.setter
    def delta(self, x):
        self.__delta = x
        for k in range(Ns):
            self.__Henergy[k*N:(k+1)*N,k*N:(k+1)*N] = self.__tan_sug(k)
    
    @property
    def FE(self):
        return self.__FE
    @FE.setter
    def FE(self, x):
        self.__FE = x
        self.__HJT[0*N:1*N,0*N:1*N] = self.__jahn_teler(0, False)
        self.__HJT[1*N:2*N,1*N:2*N] = self.__jahn_teler(2, True)
        self.__HJT[2*N:3*N,2*N:3*N] = self.__jahn_teler(0, False)
        self.__HJT[3*N:4*N,3*N:4*N] = self.__jahn_teler(2, True)
        self.__HJT[4*N:5*N,4*N:5*N] = self.__jahn_teler(1, False)
        self.__HJT[5*N:6*N,5*N:6*N] = self.__jahn_teler(0, True)
        self.__HJT[6*N:7*N,6*N:7*N] = self.__jahn_teler(1, False)
        self.__HJT[7*N:8*N,7*N:8*N] = self.__jahn_teler(0, True)
    
    @property
    def GE(self):
        return self.__GE
    @GE.setter
    def GE(self, x):
        self.__GE = x
        self.__HJT[0*N:1*N,0*N:1*N] = self.__jahn_teler(0, False)
        self.__HJT[1*N:2*N,1*N:2*N] = self.__jahn_teler(2, True)
        self.__HJT[2*N:3*N,2*N:3*N] = self.__jahn_teler(0, False)
        self.__HJT[3*N:4*N,3*N:4*N] = self.__jahn_teler(2, True)
        self.__HJT[4*N:5*N,4*N:5*N] = self.__jahn_teler(1, False)
        self.__HJT[5*N:6*N,5*N:6*N] = self.__jahn_teler(0, True)
        self.__HJT[6*N:7*N,6*N:7*N] = self.__jahn_teler(1, False)
        self.__HJT[7*N:8*N,7*N:8*N] = self.__jahn_teler(0, True)
    
    @property
    def FT(self):
        return self.__FT
    @FT.setter
    def FT(self, x):
        self.__FT = x
        self.__HJT[0*N:1*N,0*N:1*N] = self.__jahn_teler(0, False)
        self.__HJT[1*N:2*N,1*N:2*N] = self.__jahn_teler(2, True)
        self.__HJT[2*N:3*N,2*N:3*N] = self.__jahn_teler(0, False)
        self.__HJT[3*N:4*N,3*N:4*N] = self.__jahn_teler(2, True)
        self.__HJT[4*N:5*N,4*N:5*N] = self.__jahn_teler(1, False)
        self.__HJT[5*N:6*N,5*N:6*N] = self.__jahn_teler(0, True)
        self.__HJT[6*N:7*N,6*N:7*N] = self.__jahn_teler(1, False)
        self.__HJT[7*N:8*N,7*N:8*N] = self.__jahn_teler(0, True)
    
    @property
    def dth(self):
        return self.__dth
    @dth.setter
    def dth(self, x):
        self.__dth = x
        self.__HJT[0*N:1*N,0*N:1*N] = self.__jahn_teler(0, False)
        self.__HJT[1*N:2*N,1*N:2*N] = self.__jahn_teler(2, True)
        self.__HJT[2*N:3*N,2*N:3*N] = self.__jahn_teler(0, False)
        self.__HJT[3*N:4*N,3*N:4*N] = self.__jahn_teler(2, True)
        self.__HJT[4*N:5*N,4*N:5*N] = self.__jahn_teler(1, False)
        self.__HJT[5*N:6*N,5*N:6*N] = self.__jahn_teler(0, True)
        self.__HJT[6*N:7*N,6*N:7*N] = self.__jahn_teler(1, False)
        self.__HJT[7*N:8*N,7*N:8*N] = self.__jahn_teler(0, True)
    
    @property
    def xiSO(self):
        return self.__xiSO
    @xiSO.setter
    def xiSO(self, x):
        self.__xiSO = x
    
    @property
    def kent(self):
        return self.__kent
    @kent.setter
    def kent(self, x):
        self.__kent = x
        self.__rd_spin = rd_vec(8, self.__kent, self.__spin_dir)
        for k in range(Ns):
            self.__HSOC[k*N:(k+1)*N,k*N:(k+1)*N] = self.__soc(self.__rd_spin[k,:])
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def tpd(self):
        return self.__tpd
    @tpd.setter
    def tpd(self, x):
        self.__tpd = x
    
    @property
    def CT(self):
        return self.__CT
    @CT.setter
    def CT(self, x):
        self.__CT = x
        for k in range(1,Ns):
            self.__tan_sug(k)
    
    @property
    def ssd(self):
        return self.__ssd
    @ssd.setter
    def ssd(self, x):
        self.__ssd = x
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def sdd(self):
        return self.__sdd
    @sdd.setter
    def sdd(self, x):
        self.__sdd = x
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def pdd(self):
        return self.__pdd
    @pdd.setter
    def pdd(self, x):
        self.__pdd = x
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def ddd(self):
        return self.__ddd
    @ddd.setter
    def ddd(self, x):
        self.__ddd = x
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def cut(self):
        return self.__cut
    @cut.setter
    def cut(self, x):
        self.__cut = x
        self.__HhopL = self.__fill_hop(rot_eps(*self.__cut)[0])
        self.__HhopR = self.__fill_hop(rot_eps(*self.__cut)[1])
    
    @property
    def dump(self):
        return self.__dump
    @dump.setter
    def dump(self, x):
        self.__dump = x
    
    @property
    def T(self):
        return self.__T
    @T.setter
    def T(self, x):
        self.__T = x
    
    @property
    def spin_pol(self):
        return self.__spin_pol
    @spin_pol.setter
    def spin_pol(self, x):
        self.__spin_pol = x
    
    @property
    def spin_dir(self):
        return self.__rd_spin
    @spin_dir.setter
    def spin_dir(self, x):
        self.__spin_dir = x
        self.__rd_spin = rd_vec(8, kent, x)
        for k in range(Ns):
            self.__HSOC[k*N:(k+1)*N,k*N:(k+1)*N] = self.__soc(self.__rd_spin[k,:])
        
    @property
    def Nsites(self):
        return Ns
    
    ################################################################################################
    #Hamiltonians getter
    
    @property
    def ensemble(self):
        return self.__ensemble
   
    @property
    def Henergy(self):
        return self.__Henergy
   
    @property
    def HJT(self):
        return self.__HJT
   
    @property
    def HSOC(self):
        return self.__xiSO*self.__HSOC
   
    @property
    def HhopL(self):
        return self.__tpd**2/self.__CT*self.__HhopL
   
    @property
    def HhopR(self):
        return self.__tpd**2/self.__CT*self.__HhopR
    
    ################################################################################################
    #Hamiltonian construction
    
    def __tan_sug(self):
        """
        Constructs the crystal-field Hamiltonian.

        Returns
        -------
        subH : ndarray
            One-site Hamiltonian.

        """
        subH = qzeros((N,N))
        #T1g
        subH[:9,:9] += self.__delta * qidentity(9)
        #Eg
        #0
        #if n!= 0:
        #    subH += self.__CT*qidentity(N)
        
        return subH
    
    def __jahn_teler(self, n, opposite=False):
        """
        

        Parameters
        ----------
        n : int
            Axis of tetragonal elongation [0,1,2]->[z,x,y].
        opposite : bool, optional
            Deviation from tetragonal elongation is clockwise (True) or anticlockwise (False). The default is False.

        Returns
        -------
        subH : ndarray
            One-site Hamiltonian.

        """
        subH = qzeros((N,N))
        theta = n * 2*np.pi/3 + (-1)**(opposite)*self.__dth
        pauliz = qarray([[1,0],[0,-1]])
        paulix = qarray([[0,1],[1,0]])
        gellmann3 = qarray([[1,0,0],[0,-1,0],[0,0,0]])
        gellmann8 = qarray([[1,0,0],[0,1,0],[0,0,-2]])/np.sqrt(3)
        
        We = (self.__FE*np.cos(theta) + self.__GE*np.cos(2*theta))*pauliz + (self.__FE*np.sin(theta) - self.__GE*np.sin(2*theta))*paulix
        Wt = -0.5*np.sqrt(3)*self.__FT*(gellmann8*np.cos(theta) - gellmann3*np.sin(theta)) - 0.5*(self.__FE + 2*self.__GE - self.FT)*qidentity(3)
        
        subH[9:,9:] = np.kron(We,qidentity(5))
        subH[:9,:9] = np.kron(Wt,qidentity(3))
        
        #Shifting all hamiltonian for ground state at E=0
        subH += np.sqrt(self.__FE**2 + self.__GE**2 + 2*self.__FE*self.__GE*np.cos(3*theta)) * qidentity(N) #+ qidentity(N)
        
        return subH
    
    def __soc(self, site_spin_dir):
        """
        Constucts spin-orbit coupling Hamiltonian.

        Parameters
        ----------
        site_spin_dir : array-like
            Spin projection direction of the site.

        Returns
        -------
        ndarray
            One-site Hamiltonian.

        """
        # subH = qarray([[np.sqrt(6),0,0, 0,0,0,  0,0,0,  -np.sqrt(6),0,0,0,0,  0,0,np.sqrt(6),0,0],
        #                 [0,0,0, 0,0,0, -1j*np.sqrt(6),0,0,  0,-np.sqrt(3),0,0,0,  0,0,0,3*np.sqrt(2),0],
        #                 [0,0,-np.sqrt(6), 0,0,0,  0,-1j*np.sqrt(6),0,  0,0,-1,0,0,  0,0,0,0,6],
        #                 [0,0,0, -np.sqrt(6),0,0,  0,-1j*np.sqrt(6),0,  0,0,-1,0,0,  6,0,0,0,0],
        #                 [0,0,0, 0,0,0,  0,0,-1j*np.sqrt(6),  0,0,0,-np.sqrt(3),0,  0,3*np.sqrt(2),0,0,0],
        #                 [0,0,0, 0,0,np.sqrt(6), 0,0,0, 0,0,0,0,-np.sqrt(6),  0,0,np.sqrt(6),0,0],
        #                 [0,1j*np.sqrt(6),0, 0,0,0,  0,0,0,  0,1j*2*np.sqrt(3),0,0,0, 0,0,0,0,0],
        #                 [0,0,1j*np.sqrt(6), 1j*np.sqrt(6),0,0,  0,0,0,  0,0,1j*4,0,0, 0,0,0,0,0],
        #                 [0,0,0, 0,1j*np.sqrt(6),0,  0,0,0,  0,0,0,1j*2*np.sqrt(3),0, 0,0,0,0,0],
        #                 [-np.sqrt(6),0,0, 0,0,0,  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,-np.sqrt(3),0, 0,0,0,  -1j*2*np.sqrt(3),0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,-1, -1,0,0,  0,-1j*4,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,0, 0,-np.sqrt(3),0, 0,0,-1j*2*np.sqrt(3),  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,0, 0,0,-np.sqrt(6),  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,0, 6,0,0,  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,0, 0,3*np.sqrt(2),0,  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [np.sqrt(6),0,0, 0,0,np.sqrt(6),  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,3*np.sqrt(2),0, 0,0,0,  0,0,0,  0,0,0,0,0, 0,0,0,0,0],
        #                 [0,0,6, 0,0,0,  0,0,0,  0,0,0,0,0, 0,0,0,0,0]])/(2*np.sqrt(3))
        
        # Ul = qarray([[-np.sqrt(0.5), np.sqrt(0.5),0],[-1j*np.sqrt(0.5),-1j*np.sqrt(0.5),0],[0,0,1]]) @ qarray([[-1j,0,0],[0,1j,0],[0,0,1]])
        # fullUl = qidentity(N)
        # fullUl[:9,:9] = np.kron(Ul,qidentity(3))
        
        subH = qzeros((N,N))
        # m1col,m1row = np.meshgrid(np.arange(1,-1-1,-1),np.arange(1,-1-1,-1))
        # m2col,m2row = np.meshgrid(np.arange(2,-2-1,-1),np.arange(2,-2-1,-1))
        
        subH[0:3,3:6] = 0.5/1j*S1z; subH[3:6,0:3] = subH[0:3,3:6].dagg()
        subH[6:9,0:3] = 0.5*1j*S1y; subH[0:3,6:9] = subH[6:9,0:3].dagg()
        subH[6:9,3:6] = 0.5/1j*S1x; subH[3:6,6:9] = subH[6:9,3:6].dagg()
        
        m2col,m1row = np.meshgrid(np.arange(2,-2-1,-1),np.arange(1,-1-1,-1))
        subH[6:9,9:14] = -1j*np.sqrt((4-abs(m2col))/3)*(m2col==m1row); subH[9:14,6:9] = subH[6:9,9:14].dagg()
        subH[0:3,9:14] = 1j/4*np.sqrt((m2col**2+3*abs(m2col)+2)/6)*(1*(m1row==(1+m2col)) - 1*(m1row==(m2col-1))); subH[9:14,0:3] = subH[0:3,9:14].dagg()
        subH[3:6,9:14] = 1/4*np.sqrt((m2col**2+3*abs(m2col)+2)/6)*(1*(m1row==(1+m2col)) + 1*(m1row==(m2col-1))); subH[9:14,3:6] = subH[3:6,9:14].dagg()
        
        subH[0:3,14:19] = 1j/4*np.sqrt((m2col**2+3*abs(m2col)+2)/2)*(1*(m1row==(1+m2col)) - 1*(m1row==(m2col-1))); subH[14:19,0:3] = subH[0:3,14:19].dagg()
        subH[3:6,14:19] = -1/4*np.sqrt((m2col**2+3*abs(m2col)+2)/2)*(1*(m1row==(1+m2col)) + 1*(m1row==(m2col-1))); subH[14:19,3:6] = subH[3:6,14:19].dagg()
        
        U1 = spin_rotation(1,*site_spin_dir)
        U2 = spin_rotation(2,*site_spin_dir)
        fullUs = qidentity(N)
        fullUs[:9,:9] = np.kron(qidentity(3),U1)
        fullUs[9:,9:] = np.kron(qidentity(2),U2)
        
        soc_transf = fullUs# @ fullUl
        
        return soc_transf @ subH @ soc_transf.dagg()
    
    def __hop(self, hop_dir, pol, sites):
        """
        Constructs the electromagnetic hopping effective hamiltonian between two sites for a given light polarization.

        Parameters
        ----------
        hop_dir : str
            Direction of hopping x,y,z.
        pol : ndarray
            Polarization vector.
        sites : tuple (int,int)
            Sites of hopping.

        Returns
        -------
        subH : ndarray
            One-site Hamiltonian.

        """
        subH = qzeros((N,N))
        
        if not ((type(pol) is qarray) and (type(pol) is np.ndarray)): #Normalizing polarization
            pol = qarray(pol)
            pol /= np.sqrt(np.sum(pol*pol.conj()))
        
        q = ["x","y","z"].index(hop_dir)
        P = np.sum(moment_tensor(self.__ssd,self.__sdd,self.__pdd,self.__ddd)[:,q,:,:]*pol[None,:,None], axis=1)
        
        # subH[9:,9:] += np.kron(P[3:],CG_2_p[:,0,None]) @ np.kron(P[3:].dagg(),CG_2_p[:,0,None].dagg()) + np.kron(P[3:],CG_2_m[:,0,None]) @ np.kron(P[3:].dagg(),CG_2_m[:,0,None].dagg())
        # subH[:9,:9] += np.kron(P[:3],CG_1_p[:,0,None]) @ np.kron(P[:3].dagg(),CG_1_p[:,0,None].dagg()) + np.kron(P[:3],CG_1_m[:,0,None]) @ np.kron(P[:3].dagg(),CG_1_m[:,0,None].dagg())
        # subH[9:,:9] += np.kron(P[3:],CG_2_p[:,0,None]) @ np.kron(P[:3].dagg(),CG_1_p[:,0,None].dagg()) + np.kron(P[3:],CG_2_m[:,0,None]) @ np.kron(P[:3].dagg(),CG_1_m[:,0,None].dagg())
        # subH[:9,9:] += np.kron(P[:3],CG_1_p[:,0,None]) @ np.kron(P[3:].dagg(),CG_2_p[:,0,None].dagg()) + np.kron(P[:3],CG_1_m[:,0,None]) @ np.kron(P[3:].dagg(),CG_2_m[:,0,None].dagg())
        
        # for K in range(4):
        #     for L in range(K,K+1):
        #         subH[9:,9:] += np.kron(P[3:],CG_2_p[:,L,None]) @ np.kron(P[3:].dagg(),CG_2_p[:,K,None].dagg()) + np.kron(P[3:],CG_2_m[:,L,None]) @ np.kron(P[3:].dagg(),CG_2_m[:,K,None].dagg())
        #         subH[:9,:9] += np.kron(P[:3],CG_1_p[:,L,None]) @ np.kron(P[:3].dagg(),CG_1_p[:,K,None].dagg()) + np.kron(P[:3],CG_1_m[:,L,None]) @ np.kron(P[:3].dagg(),CG_1_m[:,K,None].dagg())
        #         subH[9:,:9] += np.kron(P[3:],CG_2_p[:,L,None]) @ np.kron(P[:3].dagg(),CG_1_p[:,K,None].dagg()) + np.kron(P[3:],CG_2_m[:,L,None]) @ np.kron(P[:3].dagg(),CG_1_m[:,K,None].dagg())
        #         subH[:9,9:] += np.kron(P[:3],CG_1_p[:,L,None]) @ np.kron(P[3:].dagg(),CG_2_p[:,K,None].dagg()) + np.kron(P[:3],CG_1_m[:,L,None]) @ np.kron(P[3:].dagg(),CG_2_m[:,K,None].dagg())
        
        Peg = P[3:]
        Pt2g = P[:3]
        spin_miss_mat1 = spin_rot_comp(1, self.__rd_spin[sites[0]], self.__rd_spin[sites[1]])
        spin_miss_mat2 = spin_rot_comp(2, self.__rd_spin[sites[0]], self.__rd_spin[sites[1]])
        subH[9:,9:] = np.kron(Peg @ Peg.dagg(), spin_miss_mat2)
        subH[:9,:9] = np.kron(Pt2g @ Pt2g.dagg(), spin_miss_mat1)
        
        return subH
    
    def __fill_hop(self, pol):
        """
        Constructs electromagnetic hopping effective Hamiltonian for all system.

        Parameters
        ----------
        pol : ndarray
            Polarization vector.

        Returns
        -------
        subH : ndarray
            One-site Hamiltonian.

        """
        subH = qzeros((N*Ns,N*Ns))
        subH[1*N:2*N,0*N:1*N] = self.__hop("x",pol,(1,0)); subH[0*N:1*N,1*N:2*N] = subH[1*N:2*N,0*N:1*N].dagg()
        subH[2*N:3*N,1*N:2*N] = self.__hop("y",pol,(2,1)); subH[1*N:2*N,2*N:3*N] = subH[2*N:3*N,1*N:2*N].dagg()
        subH[3*N:4*N,2*N:3*N] = self.__hop("x",pol,(3,2)); subH[2*N:3*N,3*N:4*N] = subH[3*N:4*N,2*N:3*N].dagg()
        subH[0*N:1*N,3*N:4*N] = self.__hop("y",pol,(0,3)); subH[3*N:4*N,0*N:1*N] = subH[0*N:1*N,3*N:4*N].dagg()
        subH[5*N:6*N,4*N:5*N] = self.__hop("x",pol,(5,4)); subH[4*N:5*N,5*N:6*N] = subH[5*N:6*N,4*N:5*N].dagg()
        subH[6*N:7*N,5*N:6*N] = self.__hop("y",pol,(6,5)); subH[5*N:6*N,6*N:7*N] = subH[6*N:7*N,5*N:6*N].dagg()
        subH[7*N:8*N,6*N:7*N] = self.__hop("x",pol,(7,6)); subH[6*N:7*N,7*N:8*N] = subH[7*N:8*N,6*N:7*N].dagg()
        subH[4*N:5*N,7*N:8*N] = self.__hop("y",pol,(4,7)); subH[7*N:8*N,4*N:5*N] = subH[4*N:5*N,7*N:8*N].dagg()
        subH[4*N:5*N,0*N:1*N] = self.__hop("z",pol,(4,0)); subH[0*N:1*N,4*N:5*N] = subH[4*N:5*N,0*N:1*N].dagg()
        subH[5*N:6*N,1*N:2*N] = self.__hop("z",pol,(5,1)); subH[1*N:2*N,5*N:6*N] = subH[5*N:6*N,1*N:2*N].dagg()
        subH[6*N:7*N,2*N:3*N] = self.__hop("z",pol,(6,2)); subH[2*N:3*N,6*N:7*N] = subH[6*N:7*N,2*N:3*N].dagg()
        subH[7*N:8*N,3*N:4*N] = self.__hop("z",pol,(7,3)); subH[3*N:4*N,7*N:8*N] = subH[7*N:8*N,3*N:4*N].dagg()
        return subH
    
    def __momentum_matrix(self, k):
        """
        Unused
        """
        kx,ky,kz = k
        M = qzeros((kz.size,Ns*N,Ns*N))
        M[:,0*N:1*N,1*N:2*N] += 2*np.cos(kx)[:,None,None]
        M[:,1*N:2*N,2*N:3*N] += 2*np.cos(ky)[:,None,None]
        M[:,2*N:3*N,3*N:4*N] += 2*np.cos(kx)[:,None,None]
        M[:,3*N:4*N,0*N:1*N] += 2*np.cos(ky)[:,None,None]
        
        M[:,4*N:5*N,5*N:6*N] += 2*np.cos(kx)[:,None,None]
        M[:,5*N:6*N,6*N:7*N] += 2*np.cos(ky)[:,None,None]
        M[:,6*N:7*N,7*N:8*N] += 2*np.cos(kx)[:,None,None]
        M[:,7*N:8*N,4*N:5*N] += 2*np.cos(ky)[:,None,None]
        
        M[:,0*N:1*N,4*N:5*N] += 2*np.cos(kz)[:,None,None]
        M[:,1*N:2*N,5*N:6*N] += 2*np.cos(kz)[:,None,None]
        M[:,2*N:3*N,6*N:7*N] += 2*np.cos(kz)[:,None,None]
        M[:,3*N:4*N,7*N:8*N] += 2*np.cos(kz)[:,None,None]
        
        return M+M.transpose(0,2,1)
    
    ###############################################################################
    #Solving
    def solve(self, x):
        """
        Solve spectral functions f the Hamiltonian for a given frequencies

        Parameters
        ----------
        x : float, array-like
            Frequencies in eV.

        Returns
        -------
        ndarray
            Non-gyrotropic signal.
        ndarray
            Gyrotrpic signal.

        """
        print(self.__xiSO)
        self.HL = self.__Henergy + self.__HJT + self.__xiSO*self.__HSOC + 2*self.__tpd**2/self.__CT*self.__HhopL
        self.HR = self.__Henergy + self.__HJT + self.__xiSO*self.__HSOC + 2*self.__tpd**2/self.__CT*self.__HhopR
        """
        #Cell idx criterion
        #i = Nx*Ny*jz + Nx*jy + jx
        macro_dimension = dim[0]*dim[1]*dim[2]*N*Ns
        self.MacroHL = np.zeros((macro_dimension,macro_dimension), dtype=np.complex128)
        self.MacroHR = np.zeros((macro_dimension,macro_dimension), dtype=np.complex128)
        for ii in range(dim[0]):
            for jj in range(dim[1]):
                for kk in range(dim[2]):
                    cidx = dim[0]*dim[1]*kk + dim[0]*jj + ii
                    print(cidx)
                    self.MacroHL[cidx*N*Ns:(cidx+1)*N*Ns,cidx*N*Ns:(cidx+1)*N*Ns] += self.HL
                    self.MacroHR[cidx*N*Ns:(cidx+1)*N*Ns,cidx*N*Ns:(cidx+1)*N*Ns] += self.HR
                    #xhop
                    if dim[0]>1:
                        cidx_neix = dim[0]*dim[1]*kk + dim[0]*jj + (ii+1)%dim[0]
                        print("x: %i" % cidx_neix)
                        self.MacroHL[cidx*N*Ns+1*N:cidx*N*Ns+2*N,cidx_neix*N*Ns+0*N:cidx_neix*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopL[1*N:2*N,0*N:1*N]
                        self.MacroHL[cidx_neix*N*Ns+0*N:cidx_neix*N*Ns+1*N,cidx*N*Ns+1*N:cidx*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopL[0*N:1*N,1*N:2*N]
                        self.MacroHL[cidx*N*Ns+2*N:cidx*N*Ns+3*N,cidx_neix*N*Ns+3*N:cidx_neix*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopL[2*N:3*N,3*N:4*N]
                        self.MacroHL[cidx_neix*N*Ns+3*N:cidx_neix*N*Ns+4*N,cidx*N*Ns+2*N:cidx*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopL[3*N:4*N,2*N:3*N]
                        self.MacroHL[cidx*N*Ns+5*N:cidx*N*Ns+6*N,cidx_neix*N*Ns+4*N:cidx_neix*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopL[5*N:6*N,4*N:5*N]
                        self.MacroHL[cidx_neix*N*Ns+4*N:cidx_neix*N*Ns+5*N,cidx*N*Ns+5*N:cidx*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopL[4*N:5*N,5*N:6*N]
                        self.MacroHL[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neix*N*Ns+7*N:cidx_neix*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopL[6*N:7*N,7*N:8*N]
                        self.MacroHL[cidx_neix*N*Ns+7*N:cidx_neix*N*Ns+8*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopL[7*N:8*N,6*N:7*N]
                        
                        self.MacroHR[cidx*N*Ns+1*N:cidx*N*Ns+2*N,cidx_neix*N*Ns+0*N:cidx_neix*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopR[1*N:2*N,0*N:1*N]
                        self.MacroHR[cidx_neix*N*Ns+0*N:cidx_neix*N*Ns+1*N,cidx*N*Ns+1*N:cidx*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopR[0*N:1*N,1*N:2*N]
                        self.MacroHR[cidx*N*Ns+2*N:cidx*N*Ns+3*N,cidx_neix*N*Ns+3*N:cidx_neix*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopR[2*N:3*N,3*N:4*N]
                        self.MacroHR[cidx_neix*N*Ns+3*N:cidx_neix*N*Ns+4*N,cidx*N*Ns+2*N:cidx*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopR[3*N:4*N,2*N:3*N]
                        self.MacroHR[cidx*N*Ns+5*N:cidx*N*Ns+6*N,cidx_neix*N*Ns+4*N:cidx_neix*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopR[5*N:6*N,4*N:5*N]
                        self.MacroHR[cidx_neix*N*Ns+4*N:cidx_neix*N*Ns+5*N,cidx*N*Ns+5*N:cidx*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopR[4*N:5*N,5*N:6*N]
                        self.MacroHR[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neix*N*Ns+7*N:cidx_neix*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopR[6*N:7*N,7*N:8*N]
                        self.MacroHR[cidx_neix*N*Ns+7*N:cidx_neix*N*Ns+8*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopR[7*N:8*N,6*N:7*N]
                    #yhop
                    if dim[1]>1:
                        cidx_neiy = dim[0]*dim[1]*kk + dim[0]*((jj+1)%dim[1]) + ii
                        print("y: %i" % cidx_neiy)
                        self.MacroHL[cidx*N*Ns+3*N:cidx*N*Ns+4*N,cidx_neiy*N*Ns+0*N:cidx_neiy*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopL[3*N:4*N,0*N:1*N]
                        self.MacroHL[cidx_neiy*N*Ns+0*N:cidx_neiy*N*Ns+1*N,cidx*N*Ns+3*N:cidx*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopL[0*N:1*N,3*N:4*N]
                        self.MacroHL[cidx*N*Ns+2*N:cidx*N*Ns+3*N,cidx_neiy*N*Ns+1*N:cidx_neiy*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopL[2*N:3*N,1*N:2*N]
                        self.MacroHL[cidx_neiy*N*Ns+1*N:cidx_neiy*N*Ns+2*N,cidx*N*Ns+2*N:cidx*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopL[1*N:2*N,2*N:3*N]
                        self.MacroHL[cidx*N*Ns+7*N:cidx*N*Ns+8*N,cidx_neiy*N*Ns+4*N:cidx_neiy*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopL[7*N:8*N,4*N:5*N]
                        self.MacroHL[cidx_neiy*N*Ns+4*N:cidx_neiy*N*Ns+5*N,cidx*N*Ns+7*N:cidx*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopL[4*N:5*N,7*N:8*N]
                        self.MacroHL[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neiy*N*Ns+5*N:cidx_neiy*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopL[6*N:7*N,5*N:6*N]
                        self.MacroHL[cidx_neiy*N*Ns+5*N:cidx_neiy*N*Ns+6*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopL[5*N:6*N,6*N:7*N]
                        
                        self.MacroHR[cidx*N*Ns+3*N:cidx*N*Ns+4*N,cidx_neiy*N*Ns+0*N:cidx_neiy*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopR[3*N:4*N,0*N:1*N]
                        self.MacroHR[cidx_neiy*N*Ns+0*N:cidx_neiy*N*Ns+1*N,cidx*N*Ns+3*N:cidx*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopR[0*N:1*N,3*N:4*N]
                        self.MacroHR[cidx*N*Ns+2*N:cidx*N*Ns+3*N,cidx_neiy*N*Ns+1*N:cidx_neiy*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopR[2*N:3*N,1*N:2*N]
                        self.MacroHR[cidx_neiy*N*Ns+1*N:cidx_neiy*N*Ns+2*N,cidx*N*Ns+2*N:cidx*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopR[1*N:2*N,2*N:3*N]
                        self.MacroHR[cidx*N*Ns+7*N:cidx*N*Ns+8*N,cidx_neiy*N*Ns+4*N:cidx_neiy*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopR[7*N:8*N,4*N:5*N]
                        self.MacroHR[cidx_neiy*N*Ns+4*N:cidx_neiy*N*Ns+5*N,cidx*N*Ns+7*N:cidx*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopR[4*N:5*N,7*N:8*N]
                        self.MacroHR[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neiy*N*Ns+5*N:cidx_neiy*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopR[6*N:7*N,5*N:6*N]
                        self.MacroHR[cidx_neiy*N*Ns+5*N:cidx_neiy*N*Ns+6*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopR[5*N:6*N,6*N:7*N]
                    #zhop
                    if dim[2]>1:
                        cidx_neiz = dim[0]*dim[1]*((kk+1)%dim[2]) + dim[0]*jj + ii
                        print("z: %i" % cidx_neiz)
                        self.MacroHL[cidx*N*Ns+4*N:cidx*N*Ns+5*N,cidx_neiz*N*Ns+0*N:cidx_neiz*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopL[4*N:5*N,0*N:1*N]
                        self.MacroHL[cidx_neiz*N*Ns+0*N:cidx_neiz*N*Ns+1*N,cidx*N*Ns+4*N:cidx*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopL[0*N:1*N,4*N:5*N]
                        self.MacroHL[cidx*N*Ns+5*N:cidx*N*Ns+6*N,cidx_neiz*N*Ns+1*N:cidx_neiz*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopL[5*N:6*N,1*N:2*N]
                        self.MacroHL[cidx_neiz*N*Ns+1*N:cidx_neiz*N*Ns+2*N,cidx*N*Ns+5*N:cidx*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopL[1*N:2*N,5*N:6*N]
                        self.MacroHL[cidx*N*Ns+7*N:cidx*N*Ns+8*N,cidx_neiz*N*Ns+3*N:cidx_neiz*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopL[7*N:8*N,3*N:4*N]
                        self.MacroHL[cidx_neiz*N*Ns+3*N:cidx_neiz*N*Ns+4*N,cidx*N*Ns+7*N:cidx*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopL[3*N:4*N,7*N:8*N]
                        self.MacroHL[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neiz*N*Ns+2*N:cidx_neiz*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopL[6*N:7*N,2*N:3*N]
                        self.MacroHL[cidx_neiz*N*Ns+2*N:cidx_neiz*N*Ns+3*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopL[2*N:3*N,6*N:7*N]
                        
                        self.MacroHR[cidx*N*Ns+4*N:cidx*N*Ns+5*N,cidx_neiz*N*Ns+0*N:cidx_neiz*N*Ns+1*N] += self.__tpd**2/self.__CT*self.__HhopR[4*N:5*N,0*N:1*N]
                        self.MacroHR[cidx_neiz*N*Ns+0*N:cidx_neiz*N*Ns+1*N,cidx*N*Ns+4*N:cidx*N*Ns+5*N] += self.__tpd**2/self.__CT*self.__HhopR[0*N:1*N,4*N:5*N]
                        self.MacroHR[cidx*N*Ns+5*N:cidx*N*Ns+6*N,cidx_neiz*N*Ns+1*N:cidx_neiz*N*Ns+2*N] += self.__tpd**2/self.__CT*self.__HhopR[5*N:6*N,1*N:2*N]
                        self.MacroHR[cidx_neiz*N*Ns+1*N:cidx_neiz*N*Ns+2*N,cidx*N*Ns+5*N:cidx*N*Ns+6*N] += self.__tpd**2/self.__CT*self.__HhopR[1*N:2*N,5*N:6*N]
                        self.MacroHR[cidx*N*Ns+7*N:cidx*N*Ns+8*N,cidx_neiz*N*Ns+3*N:cidx_neiz*N*Ns+4*N] += self.__tpd**2/self.__CT*self.__HhopR[7*N:8*N,3*N:4*N]
                        self.MacroHR[cidx_neiz*N*Ns+3*N:cidx_neiz*N*Ns+4*N,cidx*N*Ns+7*N:cidx*N*Ns+8*N] += self.__tpd**2/self.__CT*self.__HhopR[3*N:4*N,7*N:8*N]
                        self.MacroHR[cidx*N*Ns+6*N:cidx*N*Ns+7*N,cidx_neiz*N*Ns+2*N:cidx_neiz*N*Ns+3*N] += self.__tpd**2/self.__CT*self.__HhopR[6*N:7*N,2*N:3*N]
                        self.MacroHR[cidx_neiz*N*Ns+2*N:cidx_neiz*N*Ns+3*N,cidx*N*Ns+6*N:cidx*N*Ns+7*N] += self.__tpd**2/self.__CT*self.__HhopR[2*N:3*N,6*N:7*N]
                    print("-----------")
        rhoL = np.sum([0.5*dos(x,self.MacroHL,range(c*N*Ns+14,c*N*Ns+19),self.__dump,self.__T,self.__spin_pol) + 0.5*dos(x,self.MacroHL,range(c*N*Ns+14+95,c*N*Ns+19+95),self.__dump,self.__T,self.__spin_pol) for c in range(dim[0]*dim[1]*dim[2])], axis=0)/(dim[0]*dim[1]*dim[2])
        rhoR = np.sum([0.5*dos(x,self.MacroHR,range(c*N*Ns+14,c*N*Ns+19),self.__dump,self.__T,self.__spin_pol) + 0.5*dos(x,self.MacroHR,range(c*N*Ns+14+95,c*N*Ns+19+95),self.__dump,self.__T,self.__spin_pol) for c in range(dim[0]*dim[1]*dim[2])], axis=0)/(dim[0]*dim[1]*dim[2])
        """
        rhoL = 0.5*dos(x,self.HL,range(14,19),self.__dump,self.__T,self.__spin_pol) + 0.5*dos(x,self.HL,range(14+95,19+95),self.__dump,self.__T,self.__spin_pol) 
        rhoR = 0.5*dos(x,self.HR,range(14,19),self.__dump,self.__T,self.__spin_pol) + 0.5*dos(x,self.HR,range(14+95,19+95),self.__dump,self.__T,self.__spin_pol) 
        
        return (rhoL+rhoR)/2, (rhoL-rhoR)/2
    
    def current_correl(self, x, eps=4.5):
        """
        Unused
        """
        self.H0 = self.__Henergy + self.__HJT + self.__xiSO*self.__HSOC
        currL = self.__fill_hop([1,0,0])#(rot_eps(*self.__cut)[0])
        currR = self.__fill_hop([0,1,0])#(rot_eps(*self.__cut)[1])
        currK = self.__fill_hop([0,0,1])#(self.__cut)
        sigma_LL = (1/x) * correl(x, self.H0, currL, currL, self.__dump, self.__T)
        sigma_RL = (1/x) * correl(x, self.H0, currR, currL, self.__dump, self.__T)
        sigma_KL = (1/x) * correl(x, self.H0, currK, currL, self.__dump, self.__T)
        sigma_LR = (1/x) * correl(x, self.H0, currL, currR, self.__dump, self.__T)
        sigma_RR = (1/x) * correl(x, self.H0, currR, currR, self.__dump, self.__T)
        sigma_KR = (1/x) * correl(x, self.H0, currK, currR, self.__dump, self.__T)
        
        return sigma_LL,sigma_RL,sigma_KL, sigma_LR,sigma_RR,sigma_KR
        
        # nL = np.sqrt(eps + 1j*sigma_L/x)
        # nR = np.sqrt(eps + 1j*sigma_R/x)
        
        # RL = abs((1-nL)/(1+nL))**2
        # RR = abs((1-nR)/(1+nR))**2
    """
    def bands(self, resol=12):
        kG = np.array([0,0,0]); kX = np.array([1,0,0]); kM = np.array([1,1,0]); kR = np.array([1,1,1])
        #G-X-R-M-G
        k = np.append(np.linspace(kG,kX,resol+1)[:-1],np.append(np.linspace(kX,kR,resol+1)[:-1],np.append(np.linspace(kR,kM,resol+1)[:-1],np.linspace(kM,kG,resol+1)))).reshape((4*resol+1,3))
        s = np.append(0, np.sqrt(np.sum((k[:-1,:]-k[1:,:])**2,axis=1)))
        plot_axis_values = np.array([np.sum(s[:k+1]) for k in range(s.size)])
        k_matrix = self.__momentum_matrix(np.pi*k.T)
        self.HL = selsf.__Henergy + self.__HJT + self.__xiSO*self.__HSOC + self.__tpd**2/self.__CT*self.__HhopL*k_matrix
        self.HR = self.__Henergy + self.__HJT + self.__xiSO*self.__HSOC + self.__tpd**2/self.__CT*self.__HhopR*k_matrix
        
        bandsL,UL = np.linalg.eigh(self.HL)
        bandsR,UR = np.linalg.eigh(self.HR)
        
        contL = UL[:-1].conj().transpose(0,2,1) @ UL[1:]
        
        return np.linalg.eigvalsh(self.HL), np.linalg.eigvalsh(self.HR), plot_axis_values
    """
