function [ dA_dphi ] = A_phi( theta_k, phi_k, shape, alpha, beta )
%A_PHI Summary of this function goes here
%   Detailed explanation goes here
[lamPlus,lamMinus] = genLam(theta_k, shape, alpha, beta);
dA_dphi = zeros(2,2,shape(1),shape(2));
for i = 1:shape(1)
    for j = 1:shape(2)
        A1=(-lamPlus(i,j)+lamMinus(i,j))*2*sin(phi_k(i,j))*cos(phi_k(i,j));
        A2=(-lamPlus(i,j)+lamMinus(i,j))*(cos(phi_k(i,j))^2-sin(phi_k(i,j))^2);
        A3=(lamPlus(i,j)-lamMinus(i,j))*2*sin(phi_k(i,j))*cos(phi_k(i,j));

        dA_dphi(:,:,i,j)=[[A1,A2];[A2,A3]];
    end
end
end

