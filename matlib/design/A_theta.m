function [ dA_dtheta ] = A_theta(theta, phi_k, alpha, beta)
%A_THETA Summary of this function goes here
%   Detailed explanation goes here
    lamPlus_theta = alpha-beta;
    lamMinus_theta = (1/alpha-1/beta)*((theta/alpha+(1-theta)/beta)^(-2));
    
    A1=lamPlus_theta*cos(phiK)^2+lamMinus_theta*sin(phiK)^2;
    A2=(-lamPlus_theta+lamMinus_theta)*sin(phiK)*cos(phiK);
    A3=lamPlus_thetap.sin(phiK)^2+lamMinus_theta*cos(phiK)^2;
  
    dA_dtheta=[[A1,A2],[A2,A3]];
end

