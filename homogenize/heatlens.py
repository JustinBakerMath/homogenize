import matplotlib.animation as anim
import matplotlib.pyplot as plt
from tqdm import tqdm

from .pdeint import *
from .design import *

plt.style.use('classic')
cmap = 'coolwarm'

class heatLens():
    def __init__(self,x,y,options={})->None:
        #DOMAIN
        self.m=len(x)
        self.n=len(y)
        self.dx = 1/(self.m+1)
        self.dy = 1/(self.n+1)
        self.domain_len = self.m*self.n
        self.domain_shape = (self.m,self.n)
        self.xx, self.yy = np.meshgrid(x,y)
        self.zeros = np.zeros(self.domain_shape)
        self.ones = np.ones(self.domain_shape)
        #BOUNDARIES
        self.x_loc = np.zeros(self.domain_shape);
        self.x_loc[0,:]=np.ones(self.m);self.x_loc[-1,:]=np.ones(self.m);
        self.y_loc = np.zeros(self.domain_shape);
        self.y_loc[:,0]=np.ones(self.n);self.y_loc[:,-1]=np.ones(self.n)
        self.interior = np.ones(self.domain_shape);
        self.interior[0,:]=np.zeros(self.m);self.interior[-1,:]=np.zeros(self.m);
        self.interior[:,0]=np.zeros(self.n);self.interior[:,-1]=np.zeros(self.n)
        #OPTIONS
        self.options=options
        self.check_options()
        #PRIMAL BOUNDARY DATA
        self.gamma = self.options['gamma']
        self.gamma_loc = self.options['gamma_loc']
        self.rho_x = self.options['rho_x']
        self.rho_xloc = self.options['rho_xloc']
        self.rho_y = self.options['rho_y']
        self.rho_yloc = self.options['rho_yloc']
        #INITIALIZE
        self.u=np.zeros(self.domain_len)
        self.p=np.zeros(self.domain_len)
        self.theta=np.zeros(self.domain_shape)#0.1*np.random.rand(self.domain_len).reshape(self.domain_shape)#
        self.phi=np.zeros(self.domain_shape)#np.pi*np.random.rand(self.domain_len).reshape(self.domain_shape)#
        self.lv=self.options['lv']
        self.vol=self.options['volume']
        #DIFFERENCES
        self.fdf = FiniteDifferenceFunctions(self.m,self.n,self.dx,self.dy)
        self.fdm = FiniteDifferenceMatrices(self.m,self.n,self.dx,self.dy)
        self.contract = lambda a,b,c : np.einsum('ijk,ijk->jk',b,
                        np.einsum('ijkl,jkl->ikl',a,c))
        self.energies = []
        #TODO: history of volume fraction
        self.VOLS = []
        self.LVS = []
        self.THETAS = []
        self.tolerance = self.options['tol']
        self.A=genA(self.theta,self.phi,[self.m,self.n],alpha=self.options['alpha'],beta=self.options['beta'])
        pass

    def iterate(self,k)->None:
        if self.options['verbose']:
            pbar = tqdm(range(k))
        else:
            pbar = range(k)
        for _ in pbar:
            if (len(self.energies)>2) and (abs(self.energies[-1]-self.energies[-2])<self.tolerance):
                break
            self._iter()
            if self.options['verbose']:
                pbar.set_description(desc="{:.1e}".format(self.energies[-1]))
        pass

    def _iter(self)->None:
        #INITIALIZE
        alpha=self.options['alpha']
        beta=self.options['beta']
        tk=self.options['tk']
        #COEFFICIENTS
        self.fdm.A = self.A
        self.fdf.A = self.A
        #PRIMAL SOLUTION VIA RESIDUAL
        primal_dirichlet_loc = self.gamma_loc
        primal_neumann_xloc = self.rho_xloc
        primal_neumann_yloc = self.rho_yloc
        primal_neumann_xbc = self.rho_x
        primal_neumann_ybc = self.rho_y
        primal_dirichlet_bc = np.zeros(self.domain_shape)
        primal_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                    primal_dirichlet_bc,primal_dirichlet_loc,
                    primal_neumann_xbc, primal_neumann_xloc,
                    primal_neumann_ybc, primal_neumann_yloc)
        u = primal_pde.MdivAgrad()
        Du = self.fdf.grad(u)     
        #ADJOINT SOLUTION VIA RESIDUAL
        adjoint_dirichlet_loc = self.gamma_loc
        adjoint_neumann_xloc = self.rho_xloc
        adjoint_neumann_yloc = self.rho_yloc
        adjoint_neumann_xbc = np.zeros(self.domain_shape)
        adjoint_neumann_ybc = np.zeros(self.domain_shape)
        adjoint_dirichlet_bc = self.gamma
        adjoint_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                    adjoint_dirichlet_bc,adjoint_dirichlet_loc,
                    adjoint_neumann_xbc, adjoint_neumann_xloc,
                    adjoint_neumann_ybc, adjoint_neumann_yloc)
        p = adjoint_pde.MdivAgrad()
        Dp = self.fdf.grad(p) 
        for i in range(self.m):
            for j in range(self.n):
                A_t=A_theta(self.theta[i,j],self.phi[i,j],alpha=alpha,beta=beta)
                rhs=self.theta[i,j]+tk*(self.lv+A_t@Du[:,i,j]@Dp[:,i,j])
                self.theta[i,j]=max([0,min([1,rhs])])
        for i in range(self.m):
            for j in range(self.n):
                A_p=A_phi(self.theta[i,j],self.phi[i,j],alpha=alpha,beta=beta)
                self.phi[i,j]=self.phi[i,j]+tk*(A_p@Du[:,i,j]@Dp[:,i,j])
        if self.options['constraint']:
            self.lv = 0 if np.sum(self.theta)*self.dx*self.dy-self.vol<0 else self.options['lv']
        self.VOLS += [np.sum(self.theta)*self.dx*self.dy]
        self.LVS += [self.lv]
        self.THETAS += [self.theta.copy()]
        self.options['tk'] = .975*tk
        #SOLUTION DATA
        self.u = u
        self.p = p
        self.Du = Du
        self.Dp = Dp
        self.ADu=self.fdf.Agrad(u)
        self.ADp=self.fdf.Agrad(p)
        self.sol=self.fdf.divAgrad(u)
        self.energy = np.sum(self.gamma*(-self.Du[0]*self.y_loc-self.Du[1]*self.x_loc))#+np.sum(p.reshape(self.domain_shape)*self.fdf.divAgrad(u))
        self.energies += [self.energy]
        self.theta_vol = np.sum(self.theta*(self.dx*self.dy))
        self.A=genA(self.theta,self.phi,[self.m,self.n],alpha=self.options['alpha'],beta=self.options['beta'])
        pass

    def check_options(self):
        assert ('gamma' in self.options) and ('gamma_loc' in self.options), 'No Dirichlet Conditions Provided.'
        assert ('rho_xloc' in self.options) or ('rho_yloc' in self.options), 'No Sources Provided.'
        if not('alpha' in self.options):
            self.options['alpha']=1
        if not('beta' in self.options):
            self.options['beta']=.1
        if not('tk' in self.options):
            self.options['tk']=.1
        if not('lv' in self.options):
            self.options['lv']=0
        if not('lv_max' in self.options):
            self.options['lv_max']=1
        if not('volume' in self.options):
            self.options['volume']=.5
        if not('tol' in self.options):
            self.options['tol']=1e-3
        if not('verbose' in self.options):
            self.options['verbose']=False
        pass

class robustHeatLens():
    def __init__(self,x,y,options={}):
        #DOMAIN
        self.m=len(x)
        self.n=len(y)
        self.dx = 1/(self.m+1)
        self.dy = 1/(self.n+1)
        self.domain_len = self.m*self.n
        self.domain_shape = (self.m,self.n)
        self.xx, self.yy = np.meshgrid(x,y)
        self.zeros = np.zeros(self.domain_shape)
        self.ones = np.ones(self.domain_shape)
        #BOUNDARIES
        self.x_loc = np.zeros(self.domain_shape);
        self.x_loc[0,:]=np.ones(self.m);self.x_loc[-1,:]=np.ones(self.m);
        self.y_loc = np.zeros(self.domain_shape);
        self.y_loc[:,0]=np.ones(self.n);self.y_loc[:,-1]=np.ones(self.n)
        self.interior = np.ones(self.domain_shape);
        self.interior[0,:]=np.zeros(self.m);self.interior[-1,:]=np.zeros(self.m);
        self.interior[:,0]=np.zeros(self.n);self.interior[:,-1]=np.zeros(self.n)
        #OPTIONS
        self.options=options
        self.check_options()
        #PRIMAL BOUNDARY DATA
        self.lam = 1
        self.gamma = self.options['gamma']
        self.gamma_loc = self.options['gamma_loc']
        self.rho_x = self.options['rho_x']
        self.rho_xloc = self.options['rho_xloc']
        self.rho_y = self.options['rho_y']
        self.rho_yloc = self.options['rho_yloc']
        #SEPARATE FLUXES
        self.rho1_xloc = self.options['rho1_xloc']
        self.rho2_xloc = self.options['rho2_xloc']
        self.rho1_yloc = self.options['rho1_yloc']
        self.rho2_yloc = self.options['rho2_yloc']
        #INITIALIZE
        self.u=np.zeros(self.domain_len)
        self.p=np.zeros(self.domain_len)
        self.theta=np.zeros(self.domain_shape)#0.1*np.random.rand(self.domain_len).reshape(self.domain_shape)#
        self.phi=np.zeros(self.domain_shape)#np.pi*np.random.rand(self.domain_len).reshape(self.domain_shape)#
        self.lv=self.options['lv']
        self.vol=self.options['volume']
        #DIFFERENCES
        self.fdf = FiniteDifferenceFunctions(self.m,self.n,self.dx,self.dy)
        self.fdm = FiniteDifferenceMatrices(self.m,self.n,self.dx,self.dy)
        self.contract = lambda a,b,c : np.einsum('ijk,ijk->jk',b,
                np.einsum('ijkl,jkl->ikl',a,c))
        #ROBUST
        self.robust = self.options['robust']
        self.energies = []
        self.energies1 = []
        self.energies2 = []
        #TODO: history of volume fraction
        self.VOLS = []
        self.LVS = []
        self.THETAS = []
        self.tolerance = self.options['tol']
        self.A=genA(self.theta,self.phi,[self.m,self.n],alpha=self.options['alpha'],beta=self.options['beta'])

    def iterate(self,k):
        if self.options['verbose']:
            pbar = tqdm(range(k))
        else:
            pbar = range(k)
        for _ in pbar:
            if (len(self.energies)>2) and (abs(self.energies[-1]-self.energies[-2])<self.tolerance):
                break
            self._iter()
            if self.options['verbose']:
                pbar.set_description(desc="{:.1e}".format(self.lv))

    def _iter(self):
        #INITIALIZE
        alpha=self.options['alpha']
        beta=self.options['beta']
        tk=self.options['tk']
        #COEFFICIENTS
        self.fdm.A = self.A
        self.fdf.A = self.A
        #PRIMAL SOLUTION VIA RESIDUAL
        primal_dirichlet_loc = self.gamma_loc
        primal_neumann_xloc = self.rho_xloc
        primal_neumann_yloc = self.rho_yloc
        primal_neumann_xbc = self.rho_x
        primal_neumann_ybc = self.rho_y
        primal_dirichlet_bc = np.zeros(self.domain_shape)
        primal_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                primal_dirichlet_bc,primal_dirichlet_loc,
                primal_neumann_xbc, primal_neumann_xloc,
                primal_neumann_ybc, primal_neumann_yloc)
        u = primal_pde.MdivAgrad()
        Du = self.fdf.grad(u)
        #ADJOINT SOLUTION VIA RESIDUAL
        adjoint_dirichlet_loc = self.gamma_loc
        adjoint_neumann_xloc = self.rho_xloc
        adjoint_neumann_yloc = self.rho_yloc
        adjoint_neumann_xbc = np.zeros(self.domain_shape)
        adjoint_neumann_ybc = np.zeros(self.domain_shape)
        adjoint_dirichlet_bc = self.gamma
        adjoint_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                adjoint_dirichlet_bc,adjoint_dirichlet_loc,
                adjoint_neumann_xbc, adjoint_neumann_xloc,
                adjoint_neumann_ybc, adjoint_neumann_yloc)
        p = adjoint_pde.MdivAgrad()
        Dp = self.fdf.grad(p)
        #SEPARATE FLUXES
        primal_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                primal_dirichlet_bc,primal_dirichlet_loc,
                primal_neumann_xbc*self.rho1_xloc, primal_neumann_xloc,
                primal_neumann_ybc*self.rho1_yloc, primal_neumann_yloc)
        u1 = primal_pde.MdivAgrad()
        Du1 = self.fdf.grad(u1)
        primal_pde = PDESolver(self.m,self.n,self.fdm, self.interior,
                primal_dirichlet_bc,primal_dirichlet_loc,
                primal_neumann_xbc*self.rho2_xloc, primal_neumann_xloc,
                primal_neumann_ybc*self.rho2_yloc, primal_neumann_yloc)
        u2 = primal_pde.MdivAgrad()
        Du2 = self.fdf.grad(u2)
        energy1 = np.sum(self.gamma*(-Du1[0]*self.y_loc-Du1[1]*self.x_loc))#+np.sum(p.reshape(self.domain_shape)*self.fdf.divAgrad(u))
        energy2 = np.sum(self.gamma*(-Du2[0]*self.y_loc-Du2[1]*self.x_loc))#+np.sum(p.reshape(self.domain_shape)*self.fdf.divAgrad(u))
        if self.robust:
            self.lam = energy1/energy2
        self.rho_x = self.rho_x-self.rho_x*self.rho2_xloc+self.lam*self.rho_x*self.rho2_xloc
        self.rho_y = self.rho_y-self.rho_y*self.rho2_yloc+self.lam*self.rho_y*self.rho2_yloc
        for i in range(self.m):
            for j in range(self.n):
                A_t=A_theta(self.theta[i,j],self.phi[i,j],alpha=alpha,beta=beta)
                rhs=self.theta[i,j]+tk*(self.lv+A_t@Du[:,i,j]@Dp[:,i,j])
                self.theta[i,j]=max([0,min([1,rhs])])
        for i in range(self.m):
            for j in range(self.n):
                A_p=A_phi(self.theta[i,j],self.phi[i,j],alpha=alpha,beta=beta)
                self.phi[i,j]=self.phi[i,j]+tk*(A_p@Du[:,i,j]@Dp[:,i,j])
        if self.options['constraint']:
            #self.lv = max(0,self.lv + tk*(np.sum(self.theta)*self.dx*self.dy-self.vol))
            print(np.sum(self.theta)*self.dx*self.dy-self.vol<0,np.sum(self.theta)*self.dx*self.dy,self.vol)
            self.lv = 0 if np.sum(self.theta)*self.dx*self.dy-self.vol<0 else self.options['lv']
        #check min/max theta is 0,1
        self.VOLS += [np.sum(self.theta)*self.dx*self.dy]
        self.LVS += [self.lv]
        self.THETAS += [self.theta.copy()]
        self.options['tk'] = .975*tk
        #SOLUTION DATA
        self.u = u
        self.u1 = u1
        self.u2 = u2
        self.p = p
        self.Du = Du
        self.Du1 = Du1
        self.Du2 = Du2
        self.Dp = Dp
        self.ADu=self.fdf.Agrad(u)
        self.ADu1=self.fdf.Agrad(u1)
        self.ADu2=self.fdf.Agrad(u2)
        self.ADp=self.fdf.Agrad(p)
        self.sol=self.fdf.divAgrad(u)
        self.energy = np.sum(self.gamma*(-self.Du[0]*self.y_loc-self.Du[1]*self.x_loc))
        self.energies += [self.energy]
        self.energies1 += [energy1]
        self.energies2 += [energy2]
        self.theta_vol = np.sum(self.theta*(self.dx*self.dy))
        self.A=genA(self.theta,self.phi,[self.m,self.n],alpha=self.options['alpha'],beta=self.options['beta'])

    def check_options(self):
        assert ('gamma' in self.options) and ('gamma_loc' in self.options), 'No Dirichlet Conditions Provided.'
        assert ('rho_xloc' in self.options) or ('rho_yloc' in self.options), 'No Sources Provided.'
        assert ('rho1_xloc' in self.options) or ('rho1_yloc' in self.options), 'No Sources Provided.'
        assert ('rho2_xloc' in self.options) or ('rho2_yloc' in self.options), 'No Sources Provided.'
        if not('alpha' in self.options):
            self.options['alpha']=1
        if not('beta' in self.options):
            self.options['beta']=.1
        if not('tk' in self.options):
            self.options['tk']=.1
        if not('constraint' in self.options):
            self.options['constraint']=False
        if not('volume' in self.options):
            self.options['volume']=.5
        if not('tol' in self.options):
            self.options['tol']=.01
        if not('robust' in self.options):
            self.options['robust']=True
        if not('verbose' in self.options):
            self.options['verbose']=False
        pass

def plotDomain(lens,outDir):
    #DIRICHLET
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/dirichlet_conditions.pdf'
    plt.imshow(lens.gamma[::-1],cmap=cmap,vmin=-1,vmax=1)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    #NEUMANN
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/neumann_conditions.pdf'
    plt.imshow(lens.rho_x[::-1]+lens.rho_y[::-1],cmap=cmap,vmin=-1,vmax=1)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')

def plotSolution(lens,outDir):
    imratio = lens.domain_shape[1]/lens.domain_shape[0]
    # A NABLA U
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/A_nabla_u.pdf'
    vfield = np.divide(lens.ADu,np.fmax(1,np.linalg.norm(lens.ADu, axis=0)))
    plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # A NABLA P
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/A_nabla_p.pdf'
    Dp = np.divide(lens.ADp,np.fmax(1,np.linalg.norm(lens.ADp, axis=0)))
    plt.quiver(lens.xx[1:-1],lens.yy[1:-1],Dp[0][1:-1],Dp[1][1:-1])
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # NABLA U
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/nabla_u.pdf'
    vfield = np.divide(lens.Du,np.fmax(1,np.linalg.norm(lens.Du, axis=0)))
    plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # NABLA P
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/nabla_p.pdf'
    Dp = np.divide(lens.Dp,np.fmax(1,np.linalg.norm(lens.Dp, axis=0)))
    plt.quiver(lens.xx[1:-1],lens.yy[1:-1],Dp[0][1:-1],Dp[1][1:-1])
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # THETA
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/theta.pdf'
    plt.imshow(lens.theta.reshape(lens.domain_shape)[::-1],cmap=cmap,vmin=0,vmax=1)
    plt.colorbar(fraction=.047*imratio)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # PHI
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/phi.pdf'
    plt.imshow(lens.phi.reshape(lens.domain_shape)[::-1],cmap=cmap,vmin=-2*np.pi,vmax=2*np.pi)
    plt.colorbar(fraction=.047*imratio)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # U
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/u.pdf'
    plt.imshow(lens.u.reshape(lens.domain_shape)[::-1],cmap=cmap)
    plt.colorbar(fraction=.047*imratio)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # P
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/p.pdf'
    plt.imshow(lens.p.reshape(lens.domain_shape)[::-1],cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar(fraction=.047*imratio)
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # DOMINANT EIGS
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/eig.pdf'
    vfeild = np.zeros([2]+list(lens.domain_shape))
    for i in range(lens.domain_shape[0]):
        for j in range(lens.domain_shape[1]):
            w,v = np.linalg.eig(lens.A[...,i,j])
            # SORT
            idx = w.argsort()[::-1] 
            w = w[idx]
            v = v[:,idx]
            vfield[:,i,j] = (w[0]-w[1])*v[:,0]
    vfield = np.divide(vfield,np.fmax(1,np.linalg.norm(vfield, axis=0)))
    plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[1][1:-1],vfield[0][1:-1])
    plt.axis('off')
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    #ROBUST
    if isinstance(lens,robustHeatLens):
        # A NABLA U1
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/A_nabla_u1.pdf'
        vfield = np.divide(lens.ADu1,np.fmax(1,np.linalg.norm(lens.ADu1, axis=0)))
        plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
        plt.axis('off')
        plt.savefig(outFile,format='pdf',bbox_inches='tight')
        # A NABLA U2
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/A_nabla_u2.pdf'
        vfield = np.divide(lens.ADu2,np.fmax(1,np.linalg.norm(lens.ADu2, axis=0)))
        plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
        plt.axis('off')
        plt.savefig(outFile,format='pdf',bbox_inches='tight')
        # NABLA U1
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/nabla_u1.pdf'
        vfield = np.divide(lens.Du1,np.fmax(1,np.linalg.norm(lens.Du1, axis=0)))
        plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
        plt.axis('off')
        plt.savefig(outFile,format='pdf',bbox_inches='tight')
        # NABLA U2
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/nabla_u2.pdf'
        vfield = np.divide(lens.Du2,np.fmax(1,np.linalg.norm(lens.Du2, axis=0)))
        plt.quiver(lens.xx[1:-1],lens.yy[1:-1],vfield[0][1:-1],vfield[1][1:-1])
        plt.axis('off')
        plt.savefig(outFile,format='pdf',bbox_inches='tight')

def plotIterates(lens,outDir):
    # VOL FRAC
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/volfrac.pdf'
    plt.plot(lens.VOLS)
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    # LVS 
    plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/lvs.pdf'
    plt.plot(lens.LVS)
    plt.savefig(outFile,format='pdf',bbox_inches='tight')
    #THETA ANIM
    fig = plt.figure(figsize=(4,4),tight_layout=True)
    outFile = outDir+'/theta_anim.gif'
    ax = fig.gca()
    lines=[ax.imshow(lens.THETAS[0][::-1],cmap=cmap,vmin=0,vmax=1)]
    def theta_update(idx):
        lines[0].set_data(lens.THETAS[idx+1][::-1])
        return lines
    ani = anim.FuncAnimation(fig,theta_update,frames=len(lens.THETAS)-1,
            interval = len(lens.THETAS)-5,blit=False)
    ani.save(outFile,"PillowWriter",fps=3)
    # ENERGIES
    if isinstance(lens,robustHeatLens):
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/energy.pdf'
        plt.plot(lens.energies,'k',label='total')
        plt.plot(lens.energies1,'b',label='flux 1')
        plt.plot(lens.energies2,'r--',label='flux 2')
        plt.legend(loc=4)
        plt.savefig(outFile,format='pdf',bbox_inches='tight')
    else:
        plt.figure(figsize=(4,4),tight_layout=True)
        outFile = outDir+'/energy.pdf'
        plt.plot(lens.energies)
        plt.savefig(outFile,format='pdf',bbox_inches='tight')
