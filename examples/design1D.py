import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('./')

from lib.krot import lin_krot
from lib.design import design1D


""" KANTOROVICH RUBINSTEIN PROBLEM """
n=100
x = np.linspace(-1,1,n)
h=x[1]-x[0]
rho = np.zeros(n); rho[n//2]=3
sigma = np.zeros(n);sigma[0]=1;sigma[-1]=2
f=sigma-rho
Gc=np.ones(n);Gc[n//4:3*n//4]=2*np.ones(n//2)

options={'diff':'ForwardEuler'}
prob = lin_krot(x,f,Gc,options)
prob.minimize()

plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.figure(tight_layout=True)
plt.plot(x,prob.sol)
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Solution $\\phi(x)$',fontsize=24)
plt.savefig('./out/kr_phi.pdf',format='pdf',bbox_inches='tight')
plt.figure(tight_layout=True)
plt.plot(x,Gc,'k',label='$\\nabla c$')
plt.plot(x,prob.D1@prob.sol,'r--',label='$\\nabla \\phi$')
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Gradient Values',fontsize=24)
plt.ylim(-2.1,2.1)
plt.legend()
plt.savefig('./out/kr_grads.pdf',format='pdf',bbox_inches='tight')
plt.figure(tight_layout=True)
plt.bar(x,rho, width = 3*h, label='$\\rho(x)$',color='k')
plt.bar(x,sigma, width = 3*h, label='$\sigma(x)$',color='white')
plt.plot(x,np.abs(prob.D1@prob.sol),'r--',linewidth=2,label='$|\\nabla\\phi|$')
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Gradient Values',fontsize=24)
plt.ylim(0,3.1)
plt.legend()
plt.savefig('./out/kr.pdf',format='pdf',bbox_inches='tight')


""" AGUMENTED KR PROBLEM """
n=100
x = np.linspace(-1,1,n)
h=x[1]-x[0]
rho = np.zeros(n); rho[n//2]=3
sigma = np.zeros(n);sigma[0]=1;sigma[-1]=2
f=sigma-rho
Gc=np.ones(n);Gc[n//4:3*n//4]=2*np.ones(n//2)

options={'diff':'ForwardEuler'}
prob = design1D(x,f,Gc,options)
prob.minimize()

plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.figure(tight_layout=True)
plt.plot(x,prob.sol)
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Solution $\\phi(x)$',fontsize=24)
plt.savefig('./out/design_phi.pdf',format='pdf',bbox_inches='tight')
plt.figure(tight_layout=True)
plt.plot(x,Gc,'k',label='$\\nabla c$')
plt.plot(x,prob.D1@prob.sol,'r--',label='$\\nabla \\phi$')
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Gradient Values',fontsize=24)
plt.ylim(-2.1,2.1)
plt.legend()
plt.savefig('./out/design_grads.pdf',format='pdf',bbox_inches='tight')
plt.figure(tight_layout=True)
plt.bar(x,rho, width = 3*h, label='$\\rho(x)$',color='k')
plt.bar(x,sigma, width = 3*h, label='$\sigma(x)$',color='white')
plt.plot(x[1:-1],prob.lam[1:-1],'r--',linewidth=2,label='$\\lambda$')
plt.xlabel('Domain $(x)$',fontsize=24)
plt.ylabel('Gradient Values',fontsize=24)
plt.ylim(0,3.1)
plt.legend()
plt.savefig('./out/design.pdf',format='pdf',bbox_inches='tight')
