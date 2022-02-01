import numpy as np

# Helper Functions for 

## Central Difference
def cent_dif(vec_fn,x):
    return np.gradient(vec_fn,x)

## 2d Gradient via Central Difference
def cent_grad(f, x, y):
    return np.gradient(f,y,x)

## Gradient Ascent
def grad_ascent(point, f_grad, N, dt=1e-6):
    point = [point[j]+f_grad[j]*dt for j in range(2)]
    return point

## Poisson Solver (Neumann) 
def neumann_2d(a,b,c,d,M,N,f):
    hx=(b-a)/(N-1)
    hy=(d-c)/(M-1)
    x=np.linspace(a,b,N)
    y=np.linspace(c,d,M)
    [X,Y] = np.meshgrid(x,y,sparse=True)
    alpha=hx**2/hy**2
    dBd=2*(1+alpha)*np.ones(M)
    dBo=-1*np.ones(M-1)
    B=np.diag(dBd)+np.diag(dBo,k=1)+np.diag(dBo,k=-1)
    B[0,1]=-2
    dD=-alpha*np.ones(M)
    D=np.diag(dD)
    dC=-2*alpha*np.ones(M)
    C=np.diag(dC)
    e1=np.eye(M)
    A1=np.kron(e1,B)
    e2 = np.diag(np.ones(M-1),k=1) + np.diag(np.ones(M-1),k=-1);
    e2[0,:] = 0;
    e2[M-1,:] = 0;
    A2 = np.kron(e2,D);
    e3 = np.diag(np.ones(M-1),k=1) + np.diag(np.ones(M-1),k=-1);
    e3[1:M,0:M+1] = 0;
    A3 = np.kron(e3,C)
    A=A1+A2+A3
    F=hx**2*f
    F=F.reshape((N)*(M))
    U=np.linalg.solve(A,F)
    return U.reshape((N),(M))
