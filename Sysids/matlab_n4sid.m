function [a_est, b_est, c_est, d_est, x0, x_min, x_max, error] = matlab_n4sid(u, y, ts, nx, plot)
    % If you want to find out the best nx, you should put it in like 1:nx.
    % See MATLAB N4SID document for more details.

    disp('Successfully executed the Matlab. Run N4SID')
    data = iddata(y, u, ts);
    [sys, x0] = n4sid(data, nx);
    x0 = x0';
    a_est = sys.A;
    b_est = sys.B;
    c_est = sys.C;
    d_est = sys.D;

    result = compare(data, sys);
    error =  mean(abs(result.y - data.y));

    [~, ~, x] = sim(sys, u);
    x_max = max(x);
    x_min = min(x);

    if plot
        compare(data, result)
        pause(5);
        close all;
    end
end   