import abc
import cvxpy as cp
from lib.findiff import *
import matplotlib.pyplot as plt 

FD1D = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}

class lin_krot(metaclass=abc.ABCMeta):
    def __init__(self,x,f,Gc,options={}):
        self.options=options
        self.check_options()
        
        self.n = len(x)
        self.h = x[1]-x[0]

        self.x=x
        self.f=f
        self.Gc=Gc

    def _solve(self):
        A = FD1D[self.options['diff']](self.n, bc = 'Neumann',matrix=True)/self.h
        c=self.f
        b=self.Gc
        phi = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(c.T@phi), [A@phi<=b,-A@phi<=b])
        prob.solve()
        return phi.value

    def check_options(self):
        if not('diff' in self.options):
            self.options['diff']='ForwardEuler'
        assert self.options['diff'] in FD1D


class aug_krot(metaclass=abc.ABCMeta):
    def __init__(self,x,f,Gc,options={}):
        self.options=options
        self.check_options()
        
        self.n = len(x)
        self.h = x[1]-x[0]

        self.x=x
        self.f=f
        self.Gc=Gc

    def _solve(self):
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
