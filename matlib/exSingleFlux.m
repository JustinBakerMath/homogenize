% import robustLens.*;

L = 2;
n=10; m=12;
x = linspace(-L/2,L/2,m);
y = linspace(-L/2,L/2,n);
dx=L/(n+1); dy=L/(m+1);
gamma_1 = zeros(n,m);
gamma_1(1,1:m) = ones(1,m);
gamma_2 = zeros(n,m);
gamma_2(1:n,1) = ones(n,1);
gamma_2(1:n,m) = ones(n,1);
gamma_3 = zeros(n,m);
gamma_3(n,1:m) = ones(1,m);
gamma_0 = zeros(n,m);
gamma_0(n,m/4+1:3*m/4) = ones(1,m/2);


alpha = 1.0;
beta = 0.9;
tk = 0.01;

theta = zeros(n,m);
phi = zeros(n,m);

for i = 1:10:
    