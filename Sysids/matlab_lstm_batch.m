function [l_i_u_weights, l_i_f_weights, l_i_c_weights, l_i_o_weights, ...
          l_r_u_weights, l_r_f_weights, l_r_c_weights, l_r_o_weights, ...
          l_b_u_weights, l_b_f_weights, l_b_c_weights, l_b_o_weights, ...
          l_cell_state, l_hidden_state, n_weights, n_bias, x_min, x_max, error, loss] ...
          = matlab_lstm_batch(u, y, batch_index, node_number, learning_rate, plot_bool)

disp('MATLAB: Successfully executed the Matlab. Run LSTM fitting')
% Highly recommend to scale [u & y]
% u: [time*batch x u_dim] / y: [time*batch x y_dim]
% plot: T/F for plotting

%% Data pre-prossing 
batch_num = size(batch_index, 2) - 1;
fprintf('MATLAB: Total batch number is %i \n', batch_num)
batch_index = batch_index + 1;
for k = 1:batch_num
    batch_u{k} = u(batch_index(k):batch_index(k+1) - 1, :)';
    batch_u{k} = batch_u{k}(:, 1:end-1);
    batch_y{k} = y(batch_index(k):batch_index(k+1) - 1, :)';
    batch_y{k} = batch_y{k}(:, 2:end);
end

nu = size(batch_u{1}, 1);
ny = size(batch_y{1}, 1);


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
[net, info] = trainNetwork(batch_u, batch_y, layers, options);
net_origin = net;
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

%% Prediction
total_x_min = [];
total_x_max = [];
total_error = [];
for kk = 1:batch_num
    n = size(batch_y{kk}, 2);
    y_predict{kk} = 0*batch_y{kk};
    x = zeros(2*node_number, n);
    x(:, 1) = [l_hidden_state', l_cell_state'];
    net = net_origin;
    for k = 1:n
        [net, y_predict{kk}(:, k)] = predictAndUpdateState(net, batch_u{kk}(:, k),'ExecutionEnvironment','cpu');
        x(:, k) = [net.Layers(2).HiddenState', net.Layers(2).CellState']; 
    end
    error = mean(mean(abs(y_predict{kk} - batch_y{kk})));
    total_x_min = [total_x_min; min(x')];
    total_x_max = [total_x_max; max(x')];
    total_error = [total_error; error];
end
x_min = min(total_x_min);
x_max = max(total_x_max);
error = mean(total_error);

if plot_bool
    for kk = batch_num:batch_num  %%%
       figure; hold on;
       for k = 1:ny
           subplot(ny, 1, k); hold on;
           plot(batch_y{kk}(k, :), 'linewidth', 2);
           plot(y_predict{kk}(k, :), '-o', 'linewidth', 2)
       end
       title(strcat('y', num2str(k)));
       pause(5); 
       close all
    end
end

end