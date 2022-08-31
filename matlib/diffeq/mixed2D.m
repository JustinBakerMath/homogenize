function [ U ] = mixed2D(A,f,m,n,dx,dy,neumann,dirichlet,neumann_loc,dirichlet_loc)
%MIXED2D Summary of this function goes here
%   Detailed explanation goes here
neumann = reshape(neumann,1,m*n);
dirichlet = reshape(dirichlet,1,m*n);
neumann_loc = diag(reshape(neumann_loc,1,m*n));
dirichlet_loc = diag(reshape(dirichlet_loc,1,m*n));

interior = ones(m,n);
interior(1,:) = zeros(1,n);
interior(m,:) = zeros(1,n);
interior(:,1) = zeros(m,1);
interior(:,n) = zeros(m,1);
interior = diag(reshape(interior,1,m*n));

x = zeros(m,n);
x(:,1) = ones(m,1);
x(:,n) = ones(m,1);
x = diag(reshape(x,1,m*n));

y = zeros(m,n);
y(1,:) = ones(1,n);
y(m,:) = ones(1,n);
y = diag(reshape(y,1,m*n));

aG = Agrad(A,m,n,dx,dy);
dAg = divAgrad(A,m,n,dx,dy);

H = interior*(-dAg) + ...
    x*neumann_loc*squeeze(aG(1))+y*neumann_loc*squeeze(aG(2))+ ...
    dirichlet_loc;

F = f + neumann + dirichlet;

U = H\F';

end

