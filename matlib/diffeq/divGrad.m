function [ dG ] = divGrad(m,n,dx,dy)
%DIVGRAD Summary of this function goes here
%   Detailed explanation goes here

D = div2D(m,n,dx,dy);
G = grad2D(m,n,dx,dy);
dG = D*G;
end

