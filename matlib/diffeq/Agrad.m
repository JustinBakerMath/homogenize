function [ aG ] = Agrad(A,m,n,dx,dy)
%AGRAD Summary of this function goes here
%   Detailed explanation goes here

A = reshape(A,[2,2,m*n]);
G = grad2D(m,n,dx,dy);
aG = zeros(2,n*m,n*m);

for k = 1:m*n
   aG(1,:,k) = dot(squeeze(A(1,:,:)),squeeze(G(:,:,k)));
   aG(2,:,k) = dot(squeeze(A(2,:,:)),squeeze(G(:,:,k)));
end