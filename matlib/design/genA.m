function [ A ] = genA( theta, phi, shape, alpha, beta )
import genLam;
%GENA Summary of this function goes here
%   Detailed explanation goes here
A = zeros(2,2,shape);
lamPlus, lamMinus = genLam(theta,shape,alpha,beta);
for i = 1:shape(1)
    for j = 1:shape(2)
        A1=lamPlus(i,j)*cos(phi(i,j))^2+lamMinus(i,j)*sin(phi(i,j))^2;
        A2=(lamPlus(i,j)+lamMinus(i,j))*sin(phi(i,j))*cos(phi(i,j));
        A3=lamPlus(i,j)*sin(phi(i,j))^2+lamMinus(i,j)*cos(phi(i,j))^2;
        A(:,i,j)=[[A1,A2],[A2,A3]];
    end
end

end

