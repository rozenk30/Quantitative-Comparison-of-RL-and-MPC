function [l_i_u_weights, l_i_f_weights, l_i_c_weights, l_i_o_weights, ...
          l_r_u_weights, l_r_f_weights, l_r_c_weights, l_r_o_weights, ...
          l_b_u_weights, l_b_f_weights, l_b_c_weights, l_b_o_weights, ...
          l_cell_state, l_hidden_state, n_weights, n_bias, x_min, x_max, error, loss] ...
          = matlab_lstm(u, y, node_number, learning_rate, plot)
disp('Successfully executed the Matlab. Run LSTM fitting')
% Highly recommend to scale [u & y]
% u: N x nu / y: N x ny, where N is the number of data
% plot: T/F for plotting

%% Data pre-prossing 
n = size(u, 1);
nu = size(u, 2);
ny = size(y, 2);

%% NN setting
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
[net, info] = trainNetwork(u', y', layers, options);
error = info.TrainingRMSE;
loss = info.TrainingLoss;

%% Parameters
l_num = net.Layers(2).NumHiddenUnits;
l_i_u_weights = net.Layers(2).InputWeights(0*l_num+1:1*l_num, :);
l_i_f_weights = net.Layers(2).InputWeights(1*l_num+1:2*l_num, :);
l_i_c_weights = net.Layers(2).InputWeights(2*l_num+1:3*l_num, :);
l_i_o_weights = net.Layers(2).InputWeights(3*l_num+1:4*l_num, :);
l_r_u_weights = net.Layers(2).RecurrentWeights(0*l_num+1:1*l_num, :);
l_r_f_weights = net.Layers(2).RecurrentWeights(1*l_num+1:2*l_num, :);
l_r_c_weights = net.Layers(2).RecurrentWeights(2*l_num+1:3*l_num, :);
l_r_o_weights = net.Layers(2).RecurrentWeights(3*l_num+1:4*l_num, :);
l_b_u_weights = net.Layers(2).Bias(0*l_num+1:1*l_num, :);
l_b_f_weights = net.Layers(2).Bias(1*l_num+1:2*l_num, :);
l_b_c_weights = net.Layers(2).Bias(2*l_num+1:3*l_num, :);
l_b_o_weights = net.Layers(2).Bias(3*l_num+1:4*l_num, :);

l_hidden_state = net.Layers(2).HiddenState;
l_cell_state = net.Layers(2).CellState;

n_weights = net.Layers(3).Weights;
n_bias = net.Layers(3).Bias;

y_predict = 0*y';
x = zeros(n, 2*node_number);
x(1, :) = [l_hidden_state', l_cell_state'];
for k = 1:n
    [net, y_predict(:, k)] = predictAndUpdateState(net, u(k, :)','ExecutionEnvironment','cpu');
    x(k, :) = [net.Layers(2).HiddenState', net.Layers(2).CellState']; 
end
y_predict = y_predict';
x_max = max(x);
x_min = min(x);

error = mean(mean(abs(y_predict - y)));
%if plot
%    for k = 1:ny
%        figure; hold on;
%        plot(y(:, 1), 'linewidth', 2);
%        plot(y_predict(:, k), '-o', 'linewidth', 2)
%        title(strcat('y', num2str(k)));
%        pause(5); close all;
%    end
%end


end