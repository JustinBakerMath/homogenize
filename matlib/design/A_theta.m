function [ dA_dtheta ] = A_theta(theta_k, phi_k, shape, alpha, beta)
%A_THETA Summary of this function goes here
%   Detailed explanation goes here
    lamPlus_theta = repmat(alpha-beta,shape(1),shape(2));
    lamMinus_theta = (1/alpha-1/beta)*((theta_k/alpha+(1-theta_k)/beta).^(-2));
    dA_dtheta = zeros(2,2,shape(1),shape(2));
    for i = 1:shape(1)
        for j = 1:shape(2)
        A1=lamPlus_theta(i,j)*cos(phi_k(i,j)).^2+lamMinus_theta(i,j)*sin(phi_k(i,j)).^2;
        A2=(-lamPlus_theta(i,j)+lamMinus_theta(i,j)).*sin(phi_k(i,j))*cos(phi_k(i,j));
        A3=lamPlus_theta(i,j)*sin(phi_k(i,j)).^2+lamMinus_theta(i,j)*cos(phi_k(i,j)).^2;

        dA_dtheta(:,:,i,j)=[[A1,A2];[A2,A3]];
        end
    end
end

