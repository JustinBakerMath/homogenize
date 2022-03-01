import cvxpy as cp
from lib.findiff import *

FD = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}

class twist1D():
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
        phi = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(self.f.T@phi), [self.D1@phi<=self.Gc,-self.D1@phi<=self.Gc])
        prob.solve()
        self.sol=phi.value

    def check_options(self):
        if not('diff' in self.options):
            self.options['diff']='ForwardEuler'
        assert self.options['diff'] in FD

