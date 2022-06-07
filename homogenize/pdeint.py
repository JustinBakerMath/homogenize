from abc import ABCMeta
import numpy as np

class FiniteDifferenceFunctions(metaclass=ABCMeta):
    def __init__(self,m,n,dx,dy,A=None):
        self.A = A
        self.m = m
        self.n = n
        self.dx = dx
        self.dy = dy
    def grad(self,v):
        v = v.reshape(self.m,self.n)
        return np.array(np.gradient(v,self.dx,self.dy,axis=(1,0),edge_order=1))
    def div(self,v):
        v = v.reshape(2,self.m,self.n)
        return np.array(np.gradient(v[0],self.dx,axis=(1),edge_order=1))+np.array(np.gradient(v[1],self.dy,axis=(0),edge_order=1))
    def divGrad(self,v):
        v = v.reshape(self.m,self.n)
        return self.div(self.grad(v))
    def Agrad(self,v):
        assert (self.A is not None)
        v = v.reshape(self.m,self.n)
        mult = lambda A,v: np.einsum('ijkl,jkl->ikl',A,v)
        return mult(self.A,self.grad(v))
    def divAgrad(self,v):
        assert (self.A is not None)
        v = v.reshape(self.m,self.n)
        mult = lambda A,v: np.einsum('ijkl,jkl->ikl',A,v)
        return self.div(mult(self.A,self.grad(v)))

class FiniteDifferenceMatrices(metaclass=ABCMeta):
    def __init__(self,m,n,dx,dy,A=None):
        self.A = A
        self.m = m
        self.n = n
        self.dx = dx
        self.dy = dy
    def grad(self):
        n,m = self.n,self.m
        #X-MAT
        x_diag = np.diag(([-1]+[0]*(m-2)+[1])*(n),0)
        x_supdiag = np.diag(([1]+[.5]*(m-2)+[0])*(n-1)+([1]+[.5]*(m-2)),1)
        x_subdiag = np.diag(([-.5]*(m-2)+[-1]+[0])*(n-1)+([-.5]*(m-2)+[-1]),-1)
        x_mtrx = x_subdiag + x_diag + x_supdiag
        #Y-MAT
        y_diag = np.diag([-1]*m+[0]*m*(n-2)+[1]*m,0)
        y_supdiag = np.diag([1]*m+[.5]*m*(n-2),m)
        y_subdiag = np.diag([-.5]*m*(n-2)+[-1]*m,-m)
        y_mtrx = y_subdiag + y_diag + y_supdiag
        return np.array([x_mtrx/self.dx,y_mtrx/self.dy])
    def div(self):
        return np.moveaxis(self.grad(),0,-1)
    def divGrad(self):
        return np.tensordot(self.div(),self.grad(),axes=([2,1],[0,1]))
    def Agrad(self):
        assert (self.A is not None)
        A = self.A.reshape([2,2,self.m*self.n])
        mult = lambda A,v: np.einsum('ijk,jkl->ikl',A,v)
        return mult(A,self.grad())
    def divAgrad(self):
        assert (self.A is not None)
        A = self.A.reshape([2,2,self.m*self.n])
        mult = lambda A,v: np.einsum('ijk,jkl->ikl',A,v)
        return np.tensordot(self.div(),mult(A,self.grad()),axes=([2,1],[0,1]))


class PDESolver(metaclass=ABCMeta):
    def __init__(self,m,n,fdm,interior,dirichlet_bc,dirichlet_loc,neumann_xbc,neumann_xloc,neumann_ybc,neumann_yloc):
        self.domain_shape = (m,n)
        self.domain_len = m*n
        self.fdm = fdm
        assert np.array_equal(interior+dirichlet_loc+neumann_xloc+neumann_yloc,np.ones(neumann_yloc.shape)), "Not equivalent to ones: {}".format(np.where(interior+dirichlet_loc+neumann_xloc+neumann_yloc-np.ones(neumann_yloc.shape)!=0))
        self.interior = np.diag(interior.reshape(self.domain_len))
        self.dirichlet_bc = dirichlet_bc.reshape(self.domain_len)
        self.dirichlet_loc = np.diag(dirichlet_loc.reshape(self.domain_len))
        self.neumann_xbc = neumann_xbc.reshape(self.domain_len)
        self.neumann_xloc = np.diag(neumann_xloc.reshape(self.domain_len))
        self.neumann_ybc = neumann_ybc.reshape(self.domain_len)
        self.neumann_yloc = np.diag(neumann_yloc.reshape(self.domain_len))
    def divAgrad(self):
        pde_mtrx = (self.interior@self.fdm.divAgrad()+self.neumann_xloc@(self.fdm.Agrad()[0])+self.neumann_yloc@(self.fdm.Agrad()[1])+self.dirichlet_loc)
        return np.linalg.solve(pde_mtrx,self.neumann_xbc+self.neumann_ybc+self.dirichlet_bc)
    def MdivAgrad(self):
        pde_mtrx = (self.interior@-self.fdm.divAgrad()+self.neumann_xloc@(self.fdm.Agrad()[0])+self.neumann_yloc@(self.fdm.Agrad()[1])+self.dirichlet_loc)
        return np.linalg.solve(pde_mtrx,self.neumann_xbc+self.neumann_ybc+self.dirichlet_bc)
    def divGrad(self):
            pde_mtrx = (self.interior@self.fdm.divGrad()+self.neumann_xloc@(self.fdm.grad()[0])+self.neumann_yloc@(self.fdm.grad()[1])+self.dirichlet_loc)
            return np.linalg.solve(pde_mtrx,self.neumann_xbc+self.neumann_ybc+self.dirichlet_bc)
    def MdivGrad(self):
            pde_mtrx = (self.interior@-self.fdm.divGrad()+self.neumann_xloc@(self.fdm.grad()[0])+self.neumann_yloc@(self.fdm.grad()[1])+self.dirichlet_loc)
            return np.linalg.solve(pde_mtrx,self.neumann_xbc+self.neumann_ybc+self.dirichlet_bc)
