addpath('design')

shape = [10,12];
alpha = 1.0;
beta = 0.8;
theta = ones(shape);

[lam_plus, lam_minus] = genLam(theta,shape,alpha,beta);