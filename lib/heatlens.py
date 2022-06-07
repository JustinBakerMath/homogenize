#bin 
import cvxpy as cp
import numpy as np
from lib.findiff import D1FE,D1BE,D1CD,D2xFE,D2yFE,D2xBE,D2yBE,D2xCD,D2yCD

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

    def minimize(self):
        n=self.n
        A = FD1D[self.options['diff']](self.n, bc = 'Neumann',matrix=True)/self.h
        f=self.f
        Gc=self.Gc
        #First Step
        psi = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A.T@psi-f)))
        prob.solve()
        #Second Step
        k = abs(psi.value)/self.h
        Gphi = np.array([np.sign(np.round(psi.value[i],2))*Gc[i] for i in range(n)])
        phi = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@phi-Gphi)))
        prob.solve()
        return phi.value,k

    def check_options(self):
        if not('diff' in self.options):
            self.options['diff']='ForwardEuler'
        assert self.options['diff'] in FD1D


FD2D = {'ForwardEuler' : (D2xFE,D2yFE), 'BackwardEuler' : (D2xBE,D2yBE), 'Centered': (D2xCD,D2yCD)}

class design2D():
    def __init__(self,x,y,f,options={}):
        self.options=options
        self.check_options()

        self.n=len(x)
        self.m=len(y)
        self.domain_len = self.n*self.m
        self.domain_shape = (self.n,self.m)
        self.xx,self.yy=np.meshgrid(x,y)
        self.f=f
        self.theta=np.zeros(self.domain_len)
        self.phi=np.zeros(self.domain_len)

        Dx=FD2D[self.options['diff']][0](self.n,self.m,bc=self.options['BC'],matrix=True)
        Dy=FD2D[self.options['diff']][1](self.n,self.m,bc=self.options['BC'],matrix=True)
        self.Grad_flat=np.vstack((Dx,Dy))
        self.Div_flat=np.hstack((Dx.T,Dy.T))


        Div=np.zeros((self.domain_len,self.domain_len,2))
        Div[:,:,0]=Dx.T
        Div[:,:,1]=Dy.T
        self.Div=Div

        Grad=np.zeros((self.domain_len,2,self.domain_len))
        Grad[:,0,:]=Dx
        Grad[:,1,:]=Dy
        self.Grad=Grad

        A=self.Div_flat@self.Grad_flat
        u0=cp.Variable(self.domain_len)
        prob=cp.Problem(cp.Minimize(cp.sum_squares(A@u0-f)))
        prob.solve()
        self.u0=u0.value

    def iterate(self,k):
        for _ in range(k):
            self._iter()

    def _iter(self):

        alpha=self.options['alpha']
        beta=self.options['beta']
        lv=self.options['lv']
        tk=self.options['tk']
        Div = self.Div
        Grad = self.Grad

        A=genA(self.theta,self.phi,[self.n,self.m],alpha=alpha,beta=beta)
        B=np.tensordot(Div,A@Grad)
          
        u=cp.Variable(self.domain_len)
        prob=cp.Problem(cp.Minimize(cp.sum_squares(B@u-self.f)))
        prob.solve()
        u=u.value

        Du=(self.Grad_flat@u).reshape(2,self.domain_len)
        sol=B@u

        p=cp.Variable(self.domain_len)
        prob=cp.Problem(cp.Minimize(cp.sum_squares(B@p+2*(u-self.u0))))
        prob.solve()
        p=p.value
        Dp=(self.Grad_flat@p).reshape(2,self.domain_len)

        for idx in range(self.domain_len):
            A_t=A_theta(self.theta[idx],self.phi[idx],alpha=alpha,beta=beta)
            rhs=self.theta[idx]-tk*(lv*A_t@Du[:,idx]@Du[:,idx])
            self.theta[idx]=max([0,min([1,rhs])])

        for idx in range(self.domain_len):
            A_p=A_phi(self.theta[idx],self.phi[idx],alpha=alpha,beta=beta)
            self.phi[idx]=self.phi[idx]*tk*(A_p@Du[:,idx]@Du[:,idx])
        self.ADu=(A@Grad)@u


    def check_options(self):
        if not('diff' in self.options):
            self.options['diff']='ForwardEuler'
        if not('BC' in self.options):
            self.options['BC']='Neumann'
        if not('alpha' in self.options):
            self.options['alpha']=.01
        if not('beta' in self.options):
            self.options['beta']=1
        if not('tk' in self.options):
            self.options['tk']=.001
        if not('lv' in self.options):
            self.options['lv']=1
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
