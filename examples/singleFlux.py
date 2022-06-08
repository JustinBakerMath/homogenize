import argparse
import numpy as np
import os

from homogenize.heatlens import heatLens, plotDomain, plotSolution, plotIterates

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dout', metavar='out_dir', type=str, default='./out/singleFlux/')
parser.add_argument('--vol', metavar='volume_frac', type=float, default=0.0)
parser.add_argument('--lv', metavar='eigen_value', type=float, default=0.0)
parser.add_argument('--tk', metavar='step_size', type=float, default=0.05)
args = parser.parse_args()

if not os.path.exists(args.dout):
    os.mkdir(args.dout)

#GRID
L = 2
n,m=40,40
y=np.linspace(-L/2,L/2,n)
x=np.linspace(-L/2,L/2,m)
dx = L/(n+1)
dy = L/(m+1)
domain_len = n*m
domain_shape = (n,m)
xx,yy=np.meshgrid(x,y)
rho_x = np.zeros(domain_shape)
rho_y = np.zeros(domain_shape)
rho_y[-1,1:-1] = -np.ones(m-2)
rho_xloc = np.zeros(domain_shape)
rho_yloc = np.zeros(domain_shape)
rho_xloc[:,-1] = np.ones(n)
rho_xloc[:,0] = np.ones(n)
rho_yloc[-1,1:-1] = np.ones(n-2)
gamma_loc = np.zeros(domain_shape)
gamma_loc[0,1:-1] = np.ones(n-2)
gamma = np.zeros(domain_shape)
gamma[0,3*n//8:5*n//8]=np.ones(n//4)
options={'lv':args.lv, 'volume':args.vol, 'tk':args.tk,
	'gamma':gamma,'gamma_loc':gamma_loc,
	'rho_x':rho_x,'rho_xloc':rho_xloc,
	'rho_y':rho_y,'rho_yloc':rho_yloc,
	'verbose':True}
prob = heatLens(x,y,options)
prob.iterate(100)
#OUTPUT
plotDomain(prob,args.dout)
plotSolution(prob,args.dout)
plotIterates(prob,args.dout)
