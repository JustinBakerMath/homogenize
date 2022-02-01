#

import numpy as np

""" One Dimensional FDM """

## First Derivative Approximations
#Forward Euler
def D1FE(N, bc="Dirichlet", matrix  = False):
    if bc == "Dirichlet":
        d = [-1]*N
        dp = [0]+[1]*(N-2)
        dm = [0]*(N-1)
    elif bc == "Mixed":
        d = [-1]*(N-1)+[1]
        dp = [0]+[1]*(N-2)
        dm = [0]*(N-2)+[-1]
    elif bc == "Neumann":
        d = [-1]*(N-1)+[1]
        dp = [1]*(N-1)
        dm = [0]*(N-2)+[-1]
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')
        
    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp

#Backward Euler
def D1BE(N, bc = "Dirichlet", matrix  = False):
    if bc == "Dirichlet":
        d = [1]*(N-1)+[1]
        dp = [0]*(N-1)
        dm = [-1]*(N-2)+[0]
    elif bc == "Mixed":
        d = [1]*(N-1)+[1]
        dp = [0]*(N-1)
        dm = [-1]*(N-1)
    elif bc == "Neumann":
        d = [-1]+[1]*(N-1)
        dp = [1]+[0]*(N-2)
        dm = [-1]*(N-1)
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')
        
    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp

#Centered Difference
def D1CD(N, bc="Dirichlet", matrix  = False):
        
    if bc == "Dirichlet":
        d = [2]+[0]*(N-2)+[2]
        dp = [0]+[1]*(N-2)
        dm = [-1]*(N-2)+[0]
    elif bc == "Mixed":
        d = [2]+[0]*(N-2)+[2]
        dp = [0]+[1]*(N-2)
        dm = [-1]*(N-2)+[-2]
    elif bc == "Neumann":
        d = [-2]+[0]*(N-2)+[2]
        dp = [2]+[1]*(N-2)
        dm = [-1]*(N-2)+[-2]
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')
        
    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp

## Second Derivative Approximations
#Centered Approximation
def DD1CD(N,bc="Dirichlet", matrix  = False):
    if bc == "Dirichlet":
        d = [1]+[-2]*(N-2)+[1]
        dp = [0]+[1]*(N-2)
        dm = [1]*(N-2)+[0]
    elif bc == "Mixed":
        d = [1]+[-2]*(N-1)
        dp = [0]+[1]*(N-2)
        dm = [-1]*(N-1)
    elif bc == "Neumann":
        d = [-2]*N
        dp = [1]*(N-1)
        dm = [-1]*(N-1)
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')
        
    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp


""" Two Dimensional FDM """

## First Derivative Approximations

#Forward Euler in x
def D2xFE(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = ([1]+[-1]*(N-2)+[1])*M
        dp = (([0]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([0]*N)*M)[:-1]
    elif bc == "Mixed":
        d = ([1]+[-1]*(N-2)+[1])*M
        dp = (([0]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([0]*(N-2)+[-1]+[0])*M)[:-1]
    elif bc == "Neumann":
        d = ([-1]+[-1]*(N-2)+[1])*M
        dp = (([1]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([0]*(N-2)+[-1]+[0])*M)[:-1]
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp

#Backward Euler in x
def D2xBE(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = ([1]+[1]*(N-2)+[1])*M
        dp = (([0]+[0]*(N-2)+[0])*M)[:-1]
        dm = (([-1]*(N-2)+[0]+[0])*M)[:-1]
    elif bc == "Mixed":
        d = ([1]+[1]*(N-2)+[1])*M
        dp = (([0]+[0]*(N-2)+[0])*M)[:-1]
        dm = (([-1]*(N-2)+[-1]+[0])*M)[:-1]
    elif bc == "Neumann":
        d = ([-1]+[-1]*(N-2)+[1])*M
        dp = (([1]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([0]*(N-2)+[-1]+[0])*M)[:-1]
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)
    return dm,d,dp

#Centered Difference in x
def D2xCD(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = ([2]+[0]*(N-2)+[2])*M
        dp = (([0]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([-1]*(N-2)+[0]+[0])*M)[:-1]
    elif bc == "Mixed":
        d = ([2]+[0]*(N-2)+[2])*M
        dp = (([0]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([-1]*(N-2)+[-2]+[0])*M)[:-1]
    elif bc == "Neumann":
        d = ([-2]+[0]*(N-2)+[2])*M
        dp = (([2]+[1]*(N-2)+[0])*M)[:-1]
        dm = (([-1]*(N-2)+[-2]+[0])*M)[:-1]
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return .5*(np.diag(d)+np.diag(dp,1)+np.diag(dm,-1))
    return .5*dm,.5*d,.5*dp

#Forward Euler in y
def D2yFE(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = [1]*(N)+([-1]*(M-2))*(N)+[1]*(N)
        dp = ([0]*(N)+([1]*(M-2)*(N)))
        dm = ([0]*(N)+([0]*(M-2)*(N)))
    elif bc == "Mixed":
        d = [1]*(N)+([-1]*(M-2))*(N)+[1]*(N)
        dp = ([0]*(N)+([1]*(M-2)*(N)))
        dm = ([0]*(M-2)*(N))+[-1]*(N)
    elif bc == "Neumann":
        d = [-1]*(N)+([-1]*(M-2))*(N)+[1]*(N)
        dp = ([1]*(N)+([1]*(M-2)*(N)))
        dm = ([0]*(M-2)*(N))+[-1]*(N)
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return np.diag(d)+np.diag(dp,N)+np.diag(dm,-N)
    return dm,d,dp

#Backward Euler in y
def D2yBE(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = [1]*(N)+([1]*(M-2))*(N)+[1]*(N)
        dp = ([0]*(N)+([0]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[0]*(N)
    elif bc == "Mixed":
        d = [1]*(N)+([1]*(M-2))*(N)+[1]*(N)
        dp = ([0]*(N)+([0]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[-1]*(N)
    elif bc == "Neumann":
        d = [-1]*(N)+([1]*(M-2))*(N)+[1]*(N)
        dp = ([1]*(N)+([0]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[-1]*(N)
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return np.diag(d)+np.diag(dp,N)+np.diag(dm,-N)
    return dm,d,dp

#Centered Difference in y
def D2yCD(N,M,bc="Dirichlet",matrix=False):
    if bc == "Dirichlet":
        d = [2]*(N)+([0]*(M-2))*(N)+[2]*(N)
        dp = ([0]*(N)+([1]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[0]*(N)
    elif bc == "Mixed":
        d = [2]*(N)+([0]*(M-2))*(N)+[2]*(N)
        dp = ([0]*(N)+([1]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[-2]*(N)
    elif bc == "Neumann":
        d = [-2]*(N)+([0]*(M-2))*(N)+[2]*(N)
        dp = ([2]*(N)+([1]*(M-2)*(N)))
        dm = ([-1]*(M-2)*(N))+[-2]*(N)
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')

    if matrix:
        return .5*(np.diag(d)+np.diag(dp,N)+np.diag(dm,-N))
    return .5*dm,.5*d,.5*dp

## Laplace Operator Approximations
# Five Point Stencil
def DD2P5(N,M,bc = "Dirichlet", matrix=False):
    if bc == "Dirichlet":
        d = [-4]*N*M
        dp = (([1]*(N-1)+[0])*M)[:-1]
        dpp = ([1]*N*(M-1))
        dm = (([1]*(N-1)+[0])*M)[:-1]
        dmm = ([1]*N*(M-1))  
    elif bc == "Mixed":
        d = ([-3]+[-4]*(N-2)+[-3])+([-4]+[-4]*(N-2)+[-4])*(M-2)+([-3]+[-4]*(N-2)+[-3])
        dp = (([1]*(N-1)+[0])*M)[:-1]
        dpp = ([1]*N*(M-1))
        dm = (([1]*(N-1)+[0])*M)[:-1]
        dmm = ([1]*N*(M-1))  
    elif bc == "Neumann":
        d = ([-2]+[-3]*(N-2)+[-2])+([-3]+[-4]*(N-2)+[-3])*(M-2)+([-2]+[-3]*(N-2)+[-2])
        dp = (([1]*(N-1)+[0])*M)[:-1]
        dpp = ([1]*N*(M-1))
        dm = (([1]*(N-1)+[0])*M)[:-1]
        dmm = ([1]*N*(M-1))  
    else: raise ValueError('Unknown boundary condition. Try: Dirichlet, Mixed, Neumann')
    
    
    if matrix:
        return np.diag(d)+np.diag(dp,1)+np.diag(dm,-1)+np.diag(dpp,N)+np.diag(dmm,-N)
    return dm,d,dp

## Variable Poisson
def VP2A(n,m,lam):

    d = [1]*n*m
    dy = ([1]*(n))*(m-1)
    dx = (([1]*(n-1)+[0])*m)[:-1]
    dz = (([1]*(n-1)+[0])*(m-1))[:-1]
    
    A0 = np.diag(d)+np.diag(dy,-m)+np.diag(dx,-1)+np.diag(dz,-m-1)
    A1 = .5*(np.diag(d)+np.diag(dy,-m))
    A2 = .5*(np.diag(d)+np.diag(dx,-1))
    A3 = .5*(np.diag(dx,-1)+np.diag(dz,-m-1))
    A4 = .5*(np.diag(dy,-m)+np.diag(dz,-m-1))

    a0 = (A0@lam).reshape((n,m))[1:,1:].flatten()
    a1 = (A1@lam).reshape((n,m))[1:,1:].flatten()[1:]
    a2 = (A2@lam).reshape((n,m))[1:,1:].flatten()[m-1:]
    a3 = (A3@lam).reshape((n,m))[1:,1:].flatten()[:-1]
    a4 = (A4@lam).reshape((n,m))[1:,1:].flatten()[:-m+1]

    for i in range(m-2):
        a1[(m-2)+i*(n-1)]=0
        a3[(m-2)+i*(n-1)]=0

    return -np.diag(a0)+np.diag(a1,1)+np.diag(a2,m-1)+np.diag(a3,-1)+np.diag(a4,-m+1)

## Variable Poisson
def VP2H(n,m,lam, eps=1e-3):
    eps = eps*np.ones(n*m)
    lam = np.power(lam+eps,-1)

    d = [1]*n*m
    dy = ([1]*(n))*(m-1)
    dx = (([1]*(n-1)+[0])*m)[:-1]
    dz = (([1]*(n-1)+[0])*(m-1))[:-1]
    
    A0 = .25**2*(np.diag(d)+np.diag(dy,-m)+np.diag(dx,-1)+np.diag(dz,-m-1))
    A1 = .5*(np.diag(d)+np.diag(dy,-m))
    A2 = .5*(np.diag(d)+np.diag(dx,-1))
    A3 = .5*(np.diag(dx,-1)+np.diag(dz,-m-1))
    A4 = .5*(np.diag(dy,-m)+np.diag(dz,-m-1))

    a0 = np.power((A0@lam),-1).reshape((n,m))[1:,1:].flatten()
    a1 = np.power((A1@lam),-1).reshape((n,m))[1:,1:].flatten()[1:]
    a2 = np.power((A2@lam),-1).reshape((n,m))[1:,1:].flatten()[m-1:]
    a3 = np.power((A3@lam),-1).reshape((n,m))[1:,1:].flatten()[:-1]
    a4 = np.power((A4@lam),-1).reshape((n,m))[1:,1:].flatten()[:-m+1]


    for i in range(m-2):
        a1[(m-2)+i*(n-1)]=0
        a3[(m-2)+i*(n-1)]=0

    return -np.diag(a0)+np.diag(a1,1)+np.diag(a2,m-1)+np.diag(a3,-1)+np.diag(a4,-m+1)

def HarmAvg(A, eps=1e-3):
    n,m = A.shape

    Ai = np.power(A,-1).flatten()
    d = [1]*n*m
    dy = ([1]*(n))*(m-1)
    dx = (([1]*(n-1)+[0])*m)[:-1]
    dz = (([1]*(n-1)+[0])*(m-1))[:-1]

    d[0]=1

    A0 = np.diag(d)+np.diag(dy,-m)+np.diag(dx,-1)+np.diag(dz,-m-1)
    
    R = np.power((A0@Ai).reshape(n,m)[1:-1,1:-1]/4,-1)
    R = np.vstack((R[0], R, R[-1]))
    R = np.vstack((R.T[0], R.T, R.T[-1]))

    return R.T


def Sharpen(A, min = 1e-5, max = 2, eps =1e-5):

    n,m = A.shape

    R = A.copy()
    R = R - np.ones((n,m))

    R[np.where(R<min)] = eps
    R[np.where(R>max)] = eps
    R[np.where(R>eps)] = 1

    return R


if __name__ == "__main__":
    np.random.seed(0)
    A = .5*np.random.rand(16).reshape(4,4)
    print(A)
    print(HarmAvg(A))