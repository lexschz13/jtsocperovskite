from .imports import *
from .qarray import *


S2x = qzeros((5,5))
S2x[0,1] = 1            ; S2x[1,0] = 1
S2x[1,2] = np.sqrt(1.5) ; S2x[2,1] = np.sqrt(1.5)
S2x[2,3] = np.sqrt(1.5) ; S2x[3,2] = np.sqrt(1.5)
S2x[3,4] = 1            ; S2x[4,3] = 1

S2y = qzeros((5,5))
S2y[0,1] = -1j              ; S2y[1,0] = 1j
S2y[1,2] = -1j*np.sqrt(1.5) ; S2y[2,1] = 1j*np.sqrt(1.5)
S2y[2,3] = -1j*np.sqrt(1.5) ; S2y[3,2] = 1j*np.sqrt(1.5)
S2y[3,4] = -1j              ; S2y[4,3] = 1j

S2z = qzeros((5,5))
for k in range(5): S2z[k,k] = 2-k


def transf2(*n):
    a,b,c = n
    S2 = (a*S2x + b*S2y + c*S2z)/np.sqrt(a**2+b**2+c**2)
    vals,vects = np.linalg.eig(S2)
    vals = np.round(vals)
    U = qzeros((3,3))
    U[:,0] = vects[:,np.where(vals==2)]
    U[:,1] = vects[:,np.where(vals==1)]
    U[:,2] = vects[:,np.where(vals==0)]
    U[:,3] = vects[:,np.where(vals==-1)]
    U[:,4] = vects[:,np.where(vals==-2)]
    return U



S1x = qzeros((3,3))
S1x[0,1] = 1; S1x[1,0] = 1
S1x[1,2] = 1; S1x[2,1] = 1
S1x *= np.sqrt(0.5)

S1y = qzeros((3,3))
S1y[0,1] = -1j; S1y[1,0] = 1j
S1y[1,2] = -1j; S1y[2,1] = 1j
S1y *= np.sqrt(0.5)

S1z = qzeros((3,3))
for k in range(3): S1z[k,k] = 1-k


def transf1(*n):
    a,b,c = n
    S1 = (a*S1x + b*S1y + c*S1z)/np.sqrt(a**2+b**2+c**2)
    vals,vects = np.linalg.eig(S1)
    vals = np.round(vals)
    U = qzeros((3,3))
    U[:,0] = vects[:,np.where(vals==1)]
    U[:,1] = vects[:,np.where(vals==0)]
    U[:,2] = vects[:,np.where(vals==-1)]
    return U



S32x = qzeros((4,4))
S32x[0,1] = 0.5*np.sqrt(3); S32x[1,0] = 0.5*np.sqrt(3)
S32x[1,2] = 1             ; S32x[2,1] = 1
S32x[2,3] = 0.5*np.sqrt(3); S32x[3,2] = 0.5*np.sqrt(3)

S32y = qzeros((4,4))
S32y[0,1] = -1j*0.5*np.sqrt(3); S32y[1,0] = 1j*0.5*np.sqrt(3)
S32y[1,2] = -1j               ; S32y[2,1] = 1j
S32y[2,3] = -1j*0.5*np.sqrt(3); S32y[3,2] = 1j*0.5*np.sqrt(3)

S32z = qzeros((4,4))
for k in range(4): S32z[k,k] = 1.5-k

def transf32(*n):
    a,b,c = n
    S32 = (a*S32x + b*S32y + c*S32z)/np.sqrt(a**2+b**2+c**2)
    vals,vects = np.linalg.eig(S32)
    vals = np.round(2*vals)/2
    U = qzeros((4,4))
    U[:,0] = vects[:,np.where(vals==1.5)]
    U[:,1] = vects[:,np.where(vals==0.5)]
    U[:,2] = vects[:,np.where(vals==-0.5)]
    U[:,3] = vects[:,np.where(vals==-1.5)]
    return U



S12x = qzeros((2,2))
S12x[0,1] = 0.5; S12x[1,0] = 0.5

S12y = qzeros((2,2))
S12y[0,1] = -1j*0.5; S12y[1,0] = 1j*0.5

S12z = qzeros((2,2))
for k in range(2): S12z[k,k] = 0.5-k

def transf12(*n):
    a,b,c = n
    S12 = (a*S12x + b*S12y + c*S12z)/np.sqrt(a**2+b**2+c**2)
    vals,vects = np.linalg.eig(S12)
    vals = np.round(2*vals)/2
    U = qzeros((2,2))
    U[:,0] = vects[:,np.where(vals==0.5)]
    U[:,1] = vects[:,np.where(vals==-0.5)]
    return U



#Reduced matrix
#Basis T1g,Eg
VT1g = 2*1j*qarray([[-np.sqrt(6),-np.sqrt(7.5)],[np.sqrt(7.5),0]])

#<SM|1qS'M'>
#Named CG_SS'_q
#q=p(+1)/z(0)/m(-1)
#|M><M'|
CG_22_p = qarray([[0,np.sqrt(1/3),0,0,0],
                  [0,0,np.sqrt(1/2),0,0],
                  [0,0,0,np.sqrt(1/2),0],
                  [0,0,0,0,np.sqrt(1/3)],
                  [0,0,0,0,0]])
CG_22_z = qarray([[-np.sqrt(2/3),0,0,0,0],
                  [0,-np.sqrt(1/6),0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,np.sqrt(1/6),0],
                  [0,0,0,0,np.sqrt(2/3)]])
CG_22_m = qarray([[0,0,0,0,0],
                  [-np.sqrt(1/3),0,0,0,0],
                  [0,-np.sqrt(1/2),0,0,0],
                  [0,0,-np.sqrt(1/2),0,0],
                  [0,0,0,-np.sqrt(1/3),0]])

CG_11_p = qarray([[0,np.sqrt(1/2),0],
                  [0,0,np.sqrt(1/2)],
                  [0,0,0]])
CG_11_z = qarray([[-np.sqrt(1/2),0,0],
                  [0,0,0],
                  [0,0,np.sqrt(1/2)]])
CG_11_m = qarray([[0,0,0],
                  [-np.sqrt(1/2),0,0],
                  [0,-np.sqrt(1/2),0]])

CG_12_p = qarray([[0,0,np.sqrt(10),0,0],
                  [0,0,0,np.sqrt(3/10),0],
                  [0,0,0,0,np.sqrt(3/5)]])
CG_12_z = qarray([[0,-np.sqrt(3/10),0,0,0],
                  [0,0,-np.sqrt(2/5),0,0],
                  [0,0,0,-np.sqrt(3/10),0]])
CG_12_m = qarray([[np.sqrt(3/5),0,0,0,0],
                  [0,np.sqrt(3/10),0,0,0],
                  [0,0,np.sqrt(1/10),0,0]])

CG_21_p = np.flip(CG_12_p.T)
CG_21_z = -np.flip(CG_12_z.T)
CG_21_m = np.flip(CG_12_m.T)


def CGcartesian(*CG):
    CGp,CGz,CGm = CG
    return np.array([-np.sqrt(0.5)*(CGp-CGm),1j*np.sqrt(0.5)*(CGp+CGm),CGz]).transpose(1,0,2)



#<SM|3/2 N 1/2 m>
#Named CG_S_m
#m=p(+1/2)/m(-1/2)
#|M><N|
CG_2_p = qarray([[1,0,0,0],
                 [0,np.sqrt(0.75),0,0],
                 [0,0,np.sqrt(0.5),0],
                 [0,0,0,0.5],
                 [0,0,0,0]])
CG_2_m = qarray([[0,0,0,0],
                 [0.5,0,0,0],
                 [0,np.sqrt(0.5),0,0],
                 [0,0,np.sqrt(0.75),0],
                 [0,0,0,1]])

CG_1_p = qarray([[0,-0.5,0,0],
                 [0,0,-np.sqrt(0.5),0],
                 [0,0,0,-np.sqrt(0.75)]])
CG_1_m = qarray([[np.sqrt(0.75),0,0,0],
                 [0,np.sqrt(0.5),0,0],
                 [0,0,0.5,0]])



#<Eg g|T1g w T1g g'>
#Named CG_ET1_w
#|g><g'|
CG_ET1_x = qarray([[1/np.sqrt(6),0,0],[-np.sqrt(0.5),0,0]])
CG_ET1_y = qarray([[0,1/np.sqrt(6),0],[0,np.sqrt(0.5),0]])
CG_ET1_z = qarray([[0,0,-np.sqrt(2/3)],[0,0,0]])
CG_ET1 = qarray([CG_ET1_x,CG_ET1_y,CG_ET1_z]).transpose(1,0,2) #g,w,g'

#<T1g g|T1g w Eg g'> = <T1g g|Eg g' T1g w>
#Named CG_T1E_w
#|g><g'|
CG_T1E_x = qarray([[-0.5,np.sqrt(0.75)],[0,0],[0,0]])
CG_T1E_y = qarray([[0,0],[-0.5,-np.sqrt(0.75)],[0,0]])
CG_T1E_z = qarray([[0,0],[0,0],[1,0]])
CG_T1E = qarray([CG_T1E_x,CG_T1E_y,CG_T1E_z]).transpose(1,0,2) #g,w,g'

#g basis for Eg is u,v since here is global symmetry
#For monoelectronic orbital it is v,u

#<T1g g|T1g w T1g g'>
#Named CG_T1T1_w
#|g><g'|
CG_T1T1_x = np.sqrt(0.5)*qarray([[0,0,0],[0,0,-1],[0,1,0]])
CG_T1T1_y = np.sqrt(0.5)*qarray([[0,0,0],[0,0,-1],[0,1,0]])
CG_T1T1_z = np.sqrt(0.5)*qarray([[0,1,0],[-1,0,0],[0,0,0]])
CG_T1T1 = qarray([CG_T1T1_x,CG_T1T1_y,CG_T1T1_z]).transpose(1,0,2) #g,w,g'


def tCG(U,CG,V):
    return U.dagg() @ CG @ V
