import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('./')

from lib.misc import *

plt.style.use('classic')

n,m=401,401

x=np.linspace(-2,2,n)
y=np.linspace(-2,2,m)
xx,yy=np.meshgrid(x,y)

sig = [.5/np.pi,.4/np.pi]
mu= [0,-1.75]
rho = gauss2D(x,y,mu,sig)

sig = [.5/np.pi,.4/np.pi]
mu= [0,1.75]
gamma = gauss2D(x,y,mu,sig)

delta = np.ones_like(xx)
bar_1 = square(xx,yy,[0,0],[2,.25])
bar_2 = square(xx,yy,[1.5,0],[.1,.25])
bar_3 = square(xx,yy,[-1.5,0],[.1,.25])
delta=delta+bar_1-bar_2-bar_3

p1=[np.where(x==-1.5),np.where(y==.5)]
p2=[np.where(x==-1.5),np.where(y==-.5)]
p3=[np.where(x==1.5),np.where(y==.5)]
p4=[np.where(x==1.5),np.where(y==-.5)]
p5=[np.where(x==0),np.where(y==.5)]
p6=[np.where(x==0),np.where(y==-.5)]

plt.figure(tight_layout=True)

plt.imshow(rho-gamma-delta,origin='upper',aspect='auto')
plt.scatter(p1[0],p1[1],marker='x',color='r')
plt.scatter(p2[0],p2[1],marker='x',color='r')
plt.scatter(p3[0],p3[1],marker='x',color='r')
plt.scatter(p4[0],p4[1],marker='x',color='r')
plt.scatter(p5[0],p5[1],marker='x',color='r')
plt.scatter(p6[0],p6[1],marker='x',color='r')

plt.savefig('./out/river.pdf',format='pdf',bbox_inches='tight')


exit()

c = np.zeros(3*(n*m)**2)

for k in range(3):
  for i1 in range(n):
    for j1 in range(m):
      for i2 in range(n):
        for j2 in range(m):
          pi = np.array([x[i1],y[j1]])
          pj = np.array([x[i2],y[j2]])
          if k==0:
           c[k*n*m*n*m+i1*n*m*m+j1*n*m+i2*m+j2]=np.linalg.norm(pi-pj)
          elif k==1:
           c[k*n*m*n*m+i1*n*m*m+j1*n*m+i2*m+j2]=np.linalg.norm(pi-p1)+np.linalg.norm(p1-p2)+np.linalg.norm(p1-pj)
          elif k==2:
           c[k*n*m*n*m+i1*n*m*m+j1*n*m+i2*m+j2]=np.linalg.norm(pi-p3)+np.linalg.norm(p3-p4)+np.linalg.norm(p4-pj)

