function [a_est, b_est, c_est, d_est, x0, x_min, x_max, error] = matlab_n4sid_batch(u, y, batch_index, ts, nx, plot_bool)
    % If you want to find out the best nx, you should put it in like 1:nx.
    % See MATLAB N4SID document for more details.
    % u: [time*batch x u_dim] / y: [time*batch x y_dim]  
    disp('MATLAB: Successfully executed the Matlab. Run N4SID')
    
    batch_num = size(batch_index, 2) - 1;
    fprintf('MATLAB: Total batch number is %i \n', batch_num)
    batch_index = batch_index + 1;
    for k = 1:batch_num
        batch_data{k} = iddata(y(batch_index(k):batch_index(k+1) - 1, :), ...
                               u(batch_index(k):batch_index(k+1) - 1, :), ts);
    end

    total_data = batch_data{1};
    for k = 2:batch_num
        total_data = merge(total_data, batch_data{k});
    end
    
    [sys, x0] = n4sid(total_data, nx);
    x0 = x0';
    a_est = sys.A;
    b_est = sys.B;
    c_est = sys.C;
    d_est = sys.D;
    disp('MATLAB: n4sid done')

    result = compare(total_data, sys);
    total_error = zeros(batch_num, 1);
    for k = 1:batch_num
        total_error(k) =  mean(mean(abs(result{k}.y - total_data.y{k})));
    end
    error = mean(total_error);

    x = [];
    for k = 1:batch_num
        [~, ~, xx] = sim(sys, batch_data{k}.u);
        x = [xx; x];
    end
    x_max = max(x);
    x_min = min(x);

    if plot_bool
        for k = 1:batch_num
            compare(batch_data{k}, result{k})
            pause(5);
            close all;
        end
    end
end   