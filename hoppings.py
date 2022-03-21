from .imports import *
from .qarray import *



def moment_tensor(A,B,C,D):
    #Pchi[q,w]
    k = np.sqrt(3)
    Pu = qarray([[-0.5*A-0.5/k*B,0,0],
                 [-0.5*A+0.25/k*B+0.125*k*D,0,0],
                 [A-0.5/k*B,0,0]]) / 1j
    
    Pv = qarray([[k*0.5*A+0.5*B,0,0],
                 [-k*0.5*A+0.25*B-0.125*D,0,0],
                 [-0.25*D,0,0]]) / 1j
    
    Pz = qzeros((3,3)) / 1j
    
    Ph = 0.5*qarray([[0,0,C],[0,0,D],[0,0,C]]) / 1j
    
    Pt = 0.5*qarray([[0,C,0],[0,C,0],[0,D,0]]) / 1j
    
    #For alpha=x
    Tx = qarray([Pz,Ph,Pt,Pv,Pu])
    
    #For alpha=y
    R = qarray([[0,0,1],[1,0,0],[0,1,0]])
    Qu = R @ Pu @ np.linalg.inv(R)
    Qv = R @ Pv @ np.linalg.inv(R)
    Qh = R @ Ph @ np.linalg.inv(R)
    Qt = R @ Pt @ np.linalg.inv(R)
    Ty = qarray([Qh,Qt,Pz,-0.5*(k*Qu+Qv),-0.5*(Qu-k*Qv)])
    
    #For alpha=z
    R = qarray([[0,1,0],[0,0,1],[1,0,0]])
    Qu = R @ Pu @ np.linalg.inv(R)
    Qv = R @ Pv @ np.linalg.inv(R)
    Qh = R @ Ph @ np.linalg.inv(R)
    Qt = R @ Pt @ np.linalg.inv(R)
    Tz = qarray([Qt,Pz,Qh,0.5*(k*Qu-Qv),-0.5*(Qu+k*Qv)])
    
    #From (alpha,chi,q,w) to (chi,q,alpha,w)
    return qarray([Tx,Ty,Tz]).transpose(1,2,0,3)
