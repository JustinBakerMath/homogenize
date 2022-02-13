import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import trange

sys.path.append('./')

from lib.findiff import *
from lib.opt import *
from lib.design import *
from lib.krot import *

np.random.seed(0)

OUT_DIR='./out/side_plates/'
if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

parser=argparse.ArgumentParser(prefix_chars='-+/',description='Input Args')
parser.add_argument('--iters',type=int,default=21,help='Number of forward iterations.')
parser.add_argument('--lv',type=float,default=1,help='Lagrange Multiplier for alpha material.')
parser.add_argument('--out_interval',type=float,default=5,help='Interval to generate output.')

args,unknown=parser.parse_known_args()

"""SETUP"""
#GRID
n=m=5*10
h=1/(n-1)
dt=1

x=np.linspace(0,1,n)
y=np.linspace(0,1,m)
xx,yy=np.meshgrid(x,y)

BC='Neumann'
domain_shape=[n,m]
domain_len=n*m

#DISTRIBUTIONS
rho=np.zeros(domain_shape)
rho[0:n//2,0]=np.ones(n//2)

sigma=np.zeros(domain_shape)
sigma[-1,n//4:3*n//4]=np.ones(n//2)

f=sigma.flatten()-rho.flatten()
scale=np.sum(sigma)
#INITIALIZE
alpha=.01
beta=1
theta=np.zeros(domain_len)
phi=np.zeros(domain_len)
#DIFFERENCE OPERATORS
Dx=D2xFE(n,m,bc=BC,matrix=True)
Dy=D2yFE(n,m,bc=BC,matrix=True)

Grad_flat=np.vstack((Dx,Dy))
Div_flat=np.hstack((Dx.T,Dy.T))

Div=np.zeros((domain_len,domain_len,2))
Div[:,:,0]=Dx.T
Div[:,:,1]=Dy.T

Grad=np.zeros((domain_len,2,domain_len))
Grad[:,0,:]=Dx
Grad[:,1,:]=Dy

plt.style.use('classic')
plt.figure(tight_layout=True)
plt.imshow(f.reshape(domain_shape)[::-1],aspect='auto')
plt.savefig(OUT_DIR+'init.pdf',format='pdf',bbox_inches='tight')

"""BACKGROUND"""
A=Div_flat@Grad_flat
u0=cp.Variable(domain_len)
prob=cp.Problem(cp.Minimize(cp.sum_squares(A@u0-f)))
prob.solve()
u0=u0.value

"""FOREGROUND"""
tk=.001
for i in trange(args.iters):
  A=genA(theta,phi,[n,m],alpha=alpha,beta=beta)
  B=np.tensordot(Div,A@Grad)
  
  u=cp.Variable(domain_len)
  prob=cp.Problem(cp.Minimize(cp.sum_squares(B@u-f)))
  prob.solve()
  u=u.value

  Du=(Grad_flat@u).reshape(2,domain_len)
  sol=B@u

  p=cp.Variable(domain_len)
  prob=cp.Problem(cp.Minimize(cp.sum_squares(B@p+2*(u-u0))))
  prob.solve()
  p=p.value
  Dp=(Grad_flat@p).reshape(2,domain_len)

  for idx in range(domain_len):
    A_t=A_theta(theta[idx],phi[idx],alpha=alpha,beta=beta)
    rhs=theta[idx]-tk*(args.lv*A_t@Du[:,idx]@Du[:,idx])
    theta[idx]=max([0,min([1,rhs])])

  for idx in range(domain_len):
    A_p=A_phi(theta[idx],phi[idx],alpha=alpha,beta=beta)
    phi[idx]=phi[idx]*tk*(A_p@Du[:,idx]@Du[:,idx])

  if i%args.out_interval==0:
    plt.figure(tight_layout=True)
    plt.subplot(231)
    plt.title('$\\nabla u$')
    plt.quiver(xx,yy,Du[0],Du[1],scale=scale)
    plt.subplot(232)
    ADu=(A@Grad)@u
    plt.title('$A^*\\nabla u$')
    plt.quiver(xx,yy,ADu.T[0],ADu.T[1],scale=scale)
    plt.subplot(233)
    plt.title('Reconstructed Solution')
    plt.imshow(sol.reshape(domain_shape)[::-1],aspect='auto')
    plt.subplot(234)
    plt.title('$\\theta$')
    plt.imshow(theta.reshape(domain_shape)[::-1],vmin=0,vmax=1,aspect='auto')
    plt.subplot(235)
    plt.title('$\\phi$')
    plt.imshow(phi.reshape(domain_shape)[::-1],vmin=0,vmax=.5*np.pi,aspect='auto')
    plt.subplot(236)
    plt.title('True Solution')
    plt.imshow(f.reshape(domain_shape)[::-1],aspect='auto')
    plt.savefig(OUT_DIR+'data_'+str(i)+'.pdf',format='pdf',bbox_inches='tight')
