import numpy as np

#LAMINATES
def genLam(theta,shape,alpha=1,beta=.5):
    lamPlus=alpha*theta+beta*(np.ones(shape)-theta)
    lamMinus=np.ma.power(theta/alpha+(np.ones(shape)-theta)/beta,-1)
    return lamPlus,lamMinus

#DESIGN MATRIX
def genA(theta,phi,shape,alpha=1,beta=.1):
    A = np.zeros([2,2]+shape)
    lamPlus,lamMinus=genLam(theta,shape,alpha,beta)
    #BLOCK A
    for i in range(shape[0]):
        for j in range(shape[1]):
            A1=lamPlus[i,j]*np.cos(phi[i,j])**2+lamMinus[i,j]*np.sin(phi[i,j])**2
            A2=(lamPlus[i,j]+lamMinus[i,j])*np.sin(phi[i,j])*np.cos(phi[i,j])
            A3=lamPlus[i,j]*np.sin(phi[i,j])**2+lamMinus[i,j]*np.cos(phi[i,j])**2
            A[...,i,j]=np.array([[A1,A2],[A2,A3]])
    return A

# dlambda / dtheta
def lam_theta(theta,shape,alpha=1,beta=.5):
    n,m=shape
    lamPlus_theta=alpha-beta
    lamMinus_theta=-(1/alpha-1/beta)*((theta/alpha+(1-theta)/beta)**(-2))
    return lamPlus_theta,lamMinus_theta

# dA / dtheta
def A_theta(theta,phiK,alpha=1,beta=.1):
    lamPlus_theta=alpha-beta
    lamMinus_theta=-(1/alpha-1/beta)*((theta/alpha+(1-theta)/beta)**(-2))

    A1=lamPlus_theta*np.cos(phiK)**2+lamMinus_theta*np.sin(phiK)**2
    A2=(-lamPlus_theta+lamMinus_theta)*np.sin(phiK)*np.cos(phiK)
    A3=lamPlus_theta*np.sin(phiK)**2+lamMinus_theta*np.cos(phiK)**2
  
    A_theta=np.array([[A1,A2],[A2,A3]])
    return A_theta

# dA / dphi
def A_phi(theta,phiK,alpha=1,beta=.1):
    lamPlus=np.ma.power(theta/alpha+(1-theta)/beta,-1)
    lamMinus=alpha*theta-beta*(1-theta)

    A1=(-lamPlus+lamMinus)*2*np.sin(phiK)*np.cos(phiK)
    A2=(-lamPlus+lamMinus)*(np.cos(phiK)**2-np.sin(phiK)**2)
    A3=(lamPlus-lamMinus)*2*np.sin(phiK)*np.cos(phiK)

    A_phi=np.array([[A1,A2],[A2,A3]])
    return A_phi
