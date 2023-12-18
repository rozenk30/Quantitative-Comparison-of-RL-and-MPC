clear; close all; clc; 
Ts = 1;
N = 20;
M = 10;
rng(100);
A = [0.9 0.1; -0.2 0.8]; 
B = [0.5; 0.5];

dx = [];
du = [];
for k = 1:M
    x = zeros(2, N+1);
    u = randn(1, N+1);
    x(:, 1) = [2; 2];
    for kk = 1:N
        x(:, kk+1) = A*x(:, kk) + B*u(:, kk) + 0.5*randn(2, 1);
    end
    dx = [dx; x'];
    du = [du; u'];
    d{k} = iddata(x', u', Ts);
end

d_all = d{1};
for k = 2:M
    d_all = merge(d_all, d{k});
end

batch_index = 0:21:210;
u = du; y = dx; node_number = 10; learning_rate = 0.2; plot_bool = true;

%%
figure;
plot(d{1});

nx = 2;
sys1 = n4sid(d{1}, nx);
sys_all = n4sid(d_all, nx);

figure; 
compare(d{1}, sys1)

figure; 
compare(d_all, sys_all)

norm(sys1.A - A)
norm(sys_all.A - A)
