%% HLIP Vis

clear; clc; close all;

%% Define System
zref = 0.6;
m = 14.607895999999995;
lam = sqrt(9.81 / zref);
A = [0 1; lam ^2 0];
x0 = [-0.1; 0.60452545];
xt = [];
tt = [];
TSSP = 0.4;
for ii = 1:5
    [t, y] = ode45(@(t, x) A * x, [0, TSSP], x0);
    x0 = y(end, :);
    x0(1) = x0(1) - 0.5 * 0.4;
    xt = [xt; y];
    tt = [tt; t + 0.4 * (ii - 1)];
end


figure
subplot(2, 1, 1)
plot(tt, xt(:, 1))
subplot(2, 1, 2)
plot(tt, xt(:, 2))

%% LQR for HLIP
A = [cosh(lam*TSSP), 1/lam * sinh(lam*TSSP); lam*sinh(lam*TSSP), cosh(lam*TSSP)];
B = [-cosh(lam*TSSP); -lam*sinh(lam*TSSP)];

K = dlqr(A, B, eye(2), 1);
disp(K)

%% LQR for LLIP
A = [cosh(lam*TSSP), 1/(m * zref * lam) * sinh(lam*TSSP); m * zref * lam*sinh(lam*TSSP), cosh(lam*TSSP)];
B = [-cosh(lam*TSSP); -m * zref * lam*sinh(lam*TSSP)];

K = dlqr(A, B, eye(2), 1);
disp(K)