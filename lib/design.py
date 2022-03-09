#bin 
import cvxpy as cp
import numpy as np
from lib.findiff import D1FE,D1BE,D1CD

FD1D = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}

class design1D():
    def __init__(self,x,f,Gc,options={}):
        self.options=options
        self.check_options()

        self.x=x
        self.f=f
        self.Gc=Gc

        self.n = len(x)
        self.h = x[1]-x[0]
        self.D1 = FD1D[self.options['diff']](self.n, bc = 'Neumann',matrix=True)/self.h

    def minimize(self):
        #First Step
        lam_t = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(self.D1.T@lam_t-self.f)))
        prob.solve()
        #Second Step
        self.lam = np.abs(lam_t.value)/self.h
        Gphi = np.array([np.sign(lam_t.value[i])*self.Gc[i] for i in range(self.n)])
        phi = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(self.D1@phi-Gphi)))
        prob.solve()
        self.sol=phi.value

    def check_options(self):
        if not('diff' in self.options):
            self.options['diff']='ForwardEuler'
        assert self.options['diff'] in FD1D


"""2D ASSETS"""
def genLam(theta,shape,alpha=1,beta=.5):
    n,m = shape
    lamMinus=np.power(theta/alpha+(np.ones(n*m)-theta)/beta,-1)
    lamPlus=alpha*theta+beta*(np.ones(n*m)-theta)
    return lamMinus,lamPlus

def lam_theta(theta,shape,alpha=1,beta=.5):
  n,m=shape
  lamPlus_theta=-np.power((1/alpha-1/beta)*(theta/alpha+(np.ones(n*m)-theta)/beta),-2)
  lamMinus_theta=alpha*theta+beta*(np.ones(n*m)-theta)

def genA(theta,phi,shape,alpha=1,beta=.1):
    n,m = shape
    size=len(theta)
    A = np.zeros((size,2,2))
    lamPlus,lamMinus=genLam(theta,shape,alpha,beta)

    for idx in range(n*m):
        A1=lamPlus[idx]*np.cos(phi[idx])**2+lamMinus[idx]*np.sin(phi[idx])**2
        A2=(lamPlus[idx]+lamMinus[idx])*np.sin(phi[idx])*np.cos(phi[idx])
        A3=lamPlus[idx]*np.sin(phi[idx])**2+lamMinus[idx]*np.cos(phi[idx])**2
        
        A[idx]=np.array([[A1,A2],[A2,A3]])

    return A

def A_theta(thetaK,phiK,alpha=1,beta=.1):
  lamPlus_theta=-(1/alpha-1/beta)*((thetaK/alpha+(1-thetaK)/beta)**(-2))
  lamMinus_theta=alpha-beta

  A1=lamPlus_theta*np.cos(phiK)**2+lamMinus_theta*np.sin(phiK)**2
  A2=(lamPlus_theta+lamMinus_theta)*np.sin(phiK)*np.cos(phiK)
  A3=lamPlus_theta*np.sin(phiK)**2+lamMinus_theta*np.cos(phiK)**2
  
  A_theta=np.array([[A1,A2],[A2,A3]])
  return A_theta

def A_phi(thetaK,phiK,alpha=1,beta=.1):
  lamPlus=np.power(thetaK/alpha+(1-thetaK)/beta,-1)
  lamMinus=alpha*thetaK-beta*(1-thetaK)

  A1=(-lamPlus+lamMinus)*2*np.sin(phiK)*np.cos(phiK)
  A2=(-lamPlus+lamMinus)*(np.cos(phiK)**2-np.sin(phiK)**2)
  A3=(lamPlus-lamMinus)*2*np.sin(phiK)*np.cos(phiK)
  
  A_phi=np.array([[A1,A2],[A2,A3]])
  return A_phi
