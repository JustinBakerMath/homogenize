import cvxpy as cp
from lib.findiff import *

FD = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}

class design1D():
    def __init__(self,x,f,Gc,options={}):
        self.options=options
        self.check_options()
        diff=self.options['diff']

        self.n = len(x)
        self.h = x[1]-x[0]
        self.D1 = FD[diff](self.n, bc = 'Neumann',matrix=True)/self.h

        self.x=x
        self.f=f
        self.Gc=Gc

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
        assert self.options['diff'] in FD

