addpath('./design/')
L = 2;
n=10; m=12;
shape = [n,m];
x = linspace(-L/2,L/2,m);
y = linspace(-L/2,L/2,n);
dx=L/(n+1); dy=L/(m+1);
% q*n = 1
gamma_1 = zeros(n,m);
gamma_1(1,1:m) = ones(1,m);
q_y = zeros(n,m);
q_y(1,1:m) = ones(1,m);
% q*n = 0
gamma_2 = zeros(n,m);
gamma_2(1:n,1) = ones(n,1);
gamma_2(1:n,m) = ones(n,1);
% T = 0, rho = 0
gamma_3 = zeros(n,m);
gamma_3(n,1:m) = ones(1,m);
% rho = 1
gamma_0 = zeros(n,m);
gamma_0(n,m/4+1:3*m/4) = ones(1,m/2);


alpha = 1.0;
beta = 0.9;
tk = 0.01;

theta = zeros(n,m);
phi = zeros(n,m);

for i = 1:10

    A = genA(theta,phi,shape,alpha,beta);

    primal_dirichlet_loc = gamma_3;
    primal_dirichlet_bc = zeros(shape);
    primal_neumann_xloc = gamma_2;
    primal_neumann_yloc = gamma_1;
    primal_neumann_xbc = zeros(shape);
    primal_neuman_ybc = q_y;
    
end