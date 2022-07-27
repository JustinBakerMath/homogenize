#!/usr/bin/python3
import argparse
import numpy as np
import os

from homogenize.heatlens import robustHeatLens, plotDomain, plotSolution, plotIterates

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dout', metavar='out_dir', type=str, default='./out/splitFlux/')
parser.add_argument('--robust', action='store_true')
parser.add_argument('--constraint', action='store_true')
parser.add_argument('--scale', type=int, default=5)
parser.add_argument('--vol', metavar='volume_frac', type=float, default=1.0)
parser.add_argument('--lv', metavar='lam', type=float, default=1.0)
parser.add_argument('--tk', metavar='step_size', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.9)
args = parser.parse_args()

if not os.path.exists(args.dout):
    os.mkdir(args.dout)

#GRID
L = 2
n,m=args.scale*8,args.scale*8
y=np.linspace(-L/2,L/2,n)
x=np.linspace(-L/2,L/2,m)
dx = L/(n+1)
dy = L/(m+1)
domain_len = n*m
domain_shape = (n,m)
xx,yy=np.meshgrid(x,y)
#FLUX
rho_x = np.zeros(domain_shape)
rho_x[-m//4-1:-1,-1] = -np.ones(m//4)
rho_x[-m//4-1:-1,0] = np.ones(m//4)
rho_y = np.zeros(domain_shape)
rho1_xloc = np.zeros(domain_shape)
rho2_xloc = np.zeros(domain_shape)
rho1_yloc = np.zeros(domain_shape)
rho2_yloc = np.zeros(domain_shape)
rho_yloc = np.zeros(domain_shape)
rho1_xloc[:,-1] = np.ones(n)
rho2_xloc[:,0] = np.ones(n)
rho_xloc = rho1_xloc+rho2_xloc
rho_yloc[-1,1:-1] = np.ones(n-2)
#WINDOW
gamma_loc = np.zeros(domain_shape)
gamma_loc[0,1:-1] = np.ones(n-2)
gamma = np.zeros(domain_shape)
gamma[0,3*n//8:5*n//8]=np.ones(n//4)
#PROBLEM
options={'alpha':args.alpha,'beta':args.beta, 'constraint':args.constraint,
         'tk':args.tk, 'lv':args.lv, 'volume':args.vol,
         'gamma':gamma,'gamma_loc':gamma_loc,
         'rho_x':rho_x,'rho_xloc':rho_xloc,'rho1_xloc':rho1_xloc,'rho2_xloc':rho2_xloc,
         'rho_y':rho_y,'rho_yloc':rho_yloc,'rho1_yloc':rho1_yloc,'rho2_yloc':rho2_yloc,
         'robust':args.robust,'verbose':True}
prob = robustHeatLens(x,y,options)
prob.iterate(200)
#OUTPUT
plotDomain(prob,args.dout)
plotSolution(prob,args.dout)
plotIterates(prob,args.dout)
