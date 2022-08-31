function [lam_plus, lam_minus] = genLam(theta, shape, alpha, beta)
%GENLAM Generates eigenvalues for the design matrix
%   input:
    % theta array : characteristic function
    % shape array : dimensions
    % alpha float : properties of material 1
    % beta float : properties of material 2
    
    lam_plus = alpha*theta+beta*(ones(shape)-theta);
    lam_minus = (theta/alpha+(ones(shape)-theta)/beta).^(-1);
end

