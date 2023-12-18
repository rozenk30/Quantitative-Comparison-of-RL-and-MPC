%% Getting costs
clear; close all; format compact; clc; 

%% Plot costs
file_name = 'cstr_results_online_cost_3.txt';
file_name2 = 'cstr_results_id-online_errors_3.txt';

row_num = 10;
col_num = 16;
smooth_num = 10;

cost_file = fopen(file_name, 'r');
cost_raw = fscanf(cost_file, '%f');
fclose(cost_file);

err_file = fopen(file_name2, 'r');
err_raw = fscanf(err_file, '%f');
fclose(err_file);

costs = reshape(round(cost_raw, 4), col_num, row_num)';
errs = reshape(round(err_raw, 4), col_num, row_num)';
cost_smooth = costs;
for k = 1:row_num
    cost_smooth(k, :) = smooth(costs(k, :), smooth_num);
end

figure; hold on; 
for k = 1:row_num
    plot(cost_smooth(k, :), 'linewidth', 2); 
end
plot(0.1*ones(1, size(cost_smooth, 2)), '--k')
xlim([1, col_num]); ylim([0.2, 1.0]); 



%% Loss plot
name = 'DDPG-STACKING-STACKING-S0-60000-0-TTT';
actor1_name = strcat(name, '-actor_loss1', '.txt');
actor2_name = strcat(name, '-actor_loss2', '.txt');
critic1_name = strcat(name, '-critic_loss1', '.txt');
critic2_name = strcat(name, '-critic_loss2', '.txt');
file1ID = fopen(actor1_name,'r');
file2ID = fopen(actor2_name,'r');
file3ID = fopen(critic1_name,'r');
file4ID = fopen(critic2_name,'r');
al1 = fscanf(file1ID, '%f');
al2 = fscanf(file2ID, '%f');
cl1 = fscanf(file3ID, '%f');
cl2 = fscanf(file4ID, '%f');
fclose(file1ID);
fclose(file2ID);
fclose(file3ID);
fclose(file4ID);

x_num = size(al1, 1);
figure; 
subplot(2, 2, 1); plot(smooth(al1(20000:end), 30)); title('actor loss 1'); xlim([1, x_num])
subplot(2, 2, 2); plot(smooth(cl1(20000:end), 60)); title('critic loss 1'); xlim([1, x_num])
subplot(2, 2, 3); plot(smooth(al2(20000:end), 30)); title('actor loss 2'); xlim([1, x_num]) 
subplot(2, 2, 4); plot(smooth(cl2(20000:end), 60)); title('critic loss 2'); xlim([1, x_num])

