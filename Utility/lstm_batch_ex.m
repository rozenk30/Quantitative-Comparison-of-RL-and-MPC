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
    uu{k} = u;
    yy{k} = x;
end
y = x;
batch_index = 0:21:210;
%%
nu = 1;
ny = 2;
node_number = 10;
learning_rate = 0.02;
layers = [sequenceInputLayer(nu)
          lstmLayer(node_number)
          fullyConnectedLayer(ny)
          regressionLayer];
options = trainingOptions('adam', ...
                          'MaxEpochs', 200, ... % double(3*node_number), ...
                          'GradientThreshold',1, ...
                          'LearnRateDropPeriod',500, ...
                          'InitialLearnRate',learning_rate, ...
                          'LearnRateSchedule','piecewise', ...
                          'LearnRateDropFactor',0., ...
                          'Verbose',0, ...
                          'Plots','training-progress');
[net, info] = trainNetwork(uu, yy, layers, options);
error = info.TrainingRMSE;
loss = info.TrainingLoss;

%%
figure; 
compare(d{1}, sys1)

figure; 
compare(d_all, sys_all)

norm(sys1.A - A)
norm(sys_all.A - A)
