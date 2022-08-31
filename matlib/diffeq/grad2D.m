function [D] = grad2D( m,n,dx,dy )
%GRAD2D Summary of this function goes here
%   Detailed explanation goes here

D = zeros(2*m*n,m*n);

%% 2nd order
x_diag = diag(repmat([-1,repmat(0,1,m-2),1],1,n));
x_supdiag = diag([repmat([1,repmat(.5,1,m-2),0],1,n-1),1,repmat(.5,1,m-2)],1);
x_subdiag = diag([repmat([repmat(-.5,1,m-2),-1,0],1,n-1),repmat(-.5,1,m-2),-1],-1);
x_mtrx = x_diag+x_supdiag+x_subdiag;

y_diag = diag([repmat(-1,1,m),repmat(0,1,m*(n-2)),repmat(1,1,m)]);
y_supdiag = diag([repmat(1,1,m),repmat(.5,1,m*(n-2))],m);
y_subdiag = diag([repmat(-.5,1,m*(n-2)),repmat(-1,1,m)],-m);
y_mtrx = y_diag+y_supdiag+y_subdiag;

%% 1st order
%x_diag = diag(repmat([-1,repmat(-1,1,m-2),1],1,n));
%x_supdiag = diag([repmat([repmat(1,1,m-1),0],1,n-1),repmat(1,1,m-1)],1);
%x_subdiag = diag([repmat(0,1,m-2),-1,repmat([repmat(0,1,m-1),-1],1,n-1),],-1);
%x_mtrx = x_diag+x_supdiag+x_subdiag;

%y_diag = diag([repmat(-1,1,m*(n-1)),repmat(1,1,n)]);
%y_supdiag = diag([repmat(1,1,m*(n-1))],m);
%y_subdiag = diag([repmat(0,1,m*(n-2)),repmat(-1,1,m)],-m);
%y_mtrx = y_diag+y_supdiag+y_subdiag;

D(:,:) = 2*[x_mtrx/dx; y_mtrx/dy];
end

