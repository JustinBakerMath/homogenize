addpath('design')
addpath('diffeq')

shape = [10,12];
alpha = 1.0;
beta = 0.8;
theta = .5*ones(shape);
phi = .5*ones(shape);
%%%%%%%%%%
% genLam %
%%%%%%%%%%
[lam_plus, lam_minus] = genLam(theta,shape,alpha,beta);
% lam_plus = 1*.5 + .8*(.5) = .9
assert(isequal(lam_plus,.9*ones(shape)),'incorrect lam_plus')
% lam_minux = (.5/1+.5/.8)^-1 = .888-
assert(isequal(lam_minus,(.5/1+.5/.8)^-1 * ones(shape)),'incorrect lam_minus')

%%%%%%%%
% genA %
%%%%%%%%
A = genA(theta, phi, shape, alpha, beta);
a = [[.9*cos(.5)^2+(.5/1+.5/.8)^-1*sin(.5)^2,(.9+(.5/1+.5/.8)^-1)*sin(.5)*cos(.5)]; [(.9+(.5/1+.5/.8)^-1)*sin(.5)*cos(.5), (.5/1+.5/.8)^-1 * cos(.5)^2+.9*sin(.5)^2]];
assert(isequal(A,repmat(a,[1,1,shape(1),shape(2)])),'Design matrix not equal')

%%%%%%%%%%%%%
% dA_dtheta %
%%%%%%%%%%%%%
dA_dtheta = A_theta(theta, phi, shape, alpha, beta);
% TODO: TEST

%%%%%%%%%%%%%
% dA_dphi %
%%%%%%%%%%%%%
dA_dphi = A_phi(theta, phi, shape, alpha, beta);
% TODO: TEST

%%%%%%%%%%
% grad2D %
%%%%%%%%%%
m=25; n=25;
x = linspace(-1,1,m);
y=linspace(-1,1,n);
[xx,yy] = meshgrid(x,y);
true = xx.^2+.5*yy.^2;
fx = 2*xx;
fy = yy;
size(fx)

D = grad2D(m, n, 2/(m-1), 2/(n-1));
h = D*reshape(true',m*n,1);
hx = reshape(h(1:n*m),m,n);
hy = reshape(h(n*m+1:2*n*m),m,n);

figure
hold on
subplot(2,2,1)
pcolor(x,y,hx');
colorbar
subplot(2,2,2)
pcolor(x,y,hy');
colorbar
subplot(2,2,3)
pcolor(x,y,fx);
colorbar
subplot(2,2,4)
pcolor(x,y,fy);
colorbar
hold off
g = reshape([fx' , fy'],2*m*n,1);
u = D\g;

figure
subplot(2,1,1)
pcolor(x,y,reshape(u,m,n));
colorbar
subplot(2,1,2)
pcolor(x,y,true);
colorbar


%%%%%%%%%
% div2D %
%%%%%%%%%
m=25; n=25;
x = linspace(-1,1,m);
y=linspace(-1,1,n);
[xx,yy] = meshgrid(x,y);
true = xx.^2+.5*yy.^2;
fx = 2*xx;
fy = yy;
size(fx)
D = div2D(m, n, 2/(m-1), 2/(n-1));
size(D)
size(true)
u = D\reshape(true',n*m,1);
ux = u(1:n*m);
uy = u(n*m+1:2*n*m);
figure
subplot(2,2,1)
pcolor(x,y,reshape(ux,m,n)');
colorbar
subplot(2,2,2)
pcolor(x,y,fx)
colorbar
subplot(2,2,3)
pcolor(x,y,reshape(ux,m,n)');
colorbar
subplot(2,2,4)
pcolor(x,y,fy)
colorbar
% TODO: TEST
assert(1==0);
%%%%%%%%%%%
% divGrad %
%%%%%%%%%%%
D = divGrad(4, 4, 1, 1)
% TODO: TEST (I THINK ISSUE HERE)

m=4; n=4;
shape = [m,n];
alpha = 1.0;
beta = 0.8;
theta = .5*ones(shape);
phi = zeros(shape);

%%%%%%%%%
% Agrad %
%%%%%%%%%
A = genA(theta, phi, shape, alpha, beta);
D = Agrad(A, 4, 4, 1, 1);
% TODO: TEST (I THINK ISSUE HERE)

%%%%%%%%%%%%
% divAgrad %
%%%%%%%%%%%%
A = genA(theta, phi, shape, alpha, beta);
D = divAgrad(A, 4, 4, 1, 1);
% TODO: TEST (I THINK ISSUE HERE)


%%%%%%%%%%%
% mixed2D %
%%%%%%%%%%%

m=20; n=20;
shape = [m,n];
alpha = 1.0;
beta = .9;
theta = .9*ones(shape);
phi = zeros(shape);
A = genA(theta, phi, shape, alpha, beta);
f = zeros(1,shape(1)*shape(2));
neumann_loc = zeros(m,n);
neumann_loc(1,:) = ones(1,n);
neumann = zeros(m,n);
neumann(1,:) = -ones(1,n);
dirichlet_loc = zeros(shape(1),shape(2));
dirichlet_loc(m,:) = ones(1,n);
dirichlet_loc(:,1) = ones(m,1);
dirichlet_loc(:,n) = ones(m,1);
dirichlet = zeros(shape(1),shape(2));
dirichlet(:,n) = ones(m,1);

U =  mixed2D(A,f,m,n,1,1,neumann,dirichlet,neumann_loc,dirichlet_loc);
pcolor(linspace(0,1,m),linspace(0,1,n),reshape(U,m,n));
% TODO: TEST (I THINK ISSUE HERE)
