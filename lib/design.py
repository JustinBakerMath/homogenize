
import numpy as np

def genLam(alpha,beta,theta):
    
    n,m = theta.shape

    lamM = np.zeros((n,m))
    lamP = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            lamM[i,j] = 1/(theta[i,j]/alpha+(1-theta[i,j])/beta)
            lamP[i,j] = alpha*theta[i,j]+beta*(1-theta[i,j])

    return lamM,lamP

def genA(lamP, lamM, phi):

    n,m = phi.shape

    A = np.zeros((n,m,4))

    for i in range(n):
        for j in range(m):
            A[i,j,0] = lamP[i,j]*(np.cos(phi[i,j])**2) + lamM[i,j]*(np.sin(phi[i,j])**2)
            A[i,j,1] = (-lamP[i,j]+lamM[i,j])*np.cos(phi[i,j])*np.sin(phi[i,j])
            A[i,j,2] = (-lamP[i,j]+lamM[i,j])*np.cos(phi[i,j])*np.sin(phi[i,j])
            A[i,j,3] = lamP[i,j]*(np.sin(phi[i,j])**2) + lamM[i,j]*(np.cos(phi[i,j])**2)

    return A


def conductivity():


    # SOLVE GRAD U

    # SOLVE GRAD P


    return 0