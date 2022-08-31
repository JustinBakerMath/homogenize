function [ dAg ] = divAgrad(A,m,n,dx,dy)
%DIVMGRAD Summary of this function goes here
%   Detailed explanation goes here

aG = Agrad(A,m,n,dx,dy);
G = grad2D(m,n,dx,dy);
dAg = dot(G,permute(aG,[1 3 2]),1);
dAg = squeeze(dAg);

end

