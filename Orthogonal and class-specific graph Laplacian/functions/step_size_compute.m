function [ alpha, nonUpdate ] = step_size_compute( Z_old, Z_gradient, options, PGD, ALPHA )
%% compute step size of PGD by #Backtracking line search, Armijo?Goldstein condition#

alpha = ALPHA(end);
counter = 0;
nonUpdate = false;
while compute_obj_L(Z_old, options, PGD) - compute_obj_L(Z_old - alpha * Z_gradient, options, PGD)...
        < alpha/1e4 * norm(Z_gradient,'fro') && counter < 16
    alpha = alpha/2;
    counter = counter + 1;
    if numel(ALPHA) == 1 && counter == 16
        alpha = 0;
        fprintf('\nWARNING: No update of Z in PGD-ADMM due to alpha = 0.\n');
        nonUpdate = true;
    end
end

end