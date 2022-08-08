function [lam_plus, lam_minus] = genLam(theta, shape, alpha, beta)
%GENLAM Summary of this function goes here
%   Detailed explanation goes here
    lam_plus = alpha*theta+beta*(ones(shape)-theta);
    lam_minus = (theta/alpha+(ones(shape)-theta)/beta).^(-1);
end

